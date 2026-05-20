# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0 #Ensures that there aren't any leftover pieces 
        self.key = nn.Linear(config['n_embd'], config['n_embd'])   #Labels, used to match against Query and Value
        self.query = nn.Linear(config['n_embd'], config['n_embd']) #What are we looking for?
        self.value = nn.Linear(config['n_embd'], config['n_embd']) #Actual content of the information, like the payload
        self.attn_drop = nn.Dropout(config['dropout'])
        self.resid_drop = nn.Dropout(config['dropout'])
        self.proj = nn.Linear(config['n_embd'], config['n_embd'])
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']
        self.register_buffer("bias", torch.tril(torch.ones(config['block_size'], config['block_size']))
                                     .view(1, 1, config['block_size'], config['block_size'])) #The mask

    def forward(self, x): 
        B, T, C = x.size() #Batch, Time, Channels
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #We transpose to ensure that each head gets their isolated data

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) #Softmax formula
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) #Prevents looking into the future by replacing future things with -ve infinity
        att = torch.softmax(att, dim=-1) #Turns them into percentages (-inf becomes 0%S)
        att = self.attn_drop(att)
        y = att @ v #Applying the percentages to actual data
        y = y.transpose(1, 2).contiguous().view(B, T, C) #Glues all 4 heads back together
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config['n_embd'])
        self.mlp = nn.Sequential(
            nn.Linear(config['n_embd'], 4 * config['n_embd']), #Blows the dimensions up to 4x original to give the AI more thinking space
            nn.GELU(), #Unlike ReLU, it allows for non-zero values, making it better for complex patterns (Non-linearity as well)
            nn.Linear(4 * config['n_embd'], config['n_embd']), #Shrinks it back down
            nn.Dropout(config['dropout']),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        #WORKING ORDER: ln_1 --> Attention --> ln_2 --> MLP
        return x

class ADTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, n_layer=4, n_head=4, n_embd=128, max_timestep=1000, dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.config = {
            'n_layer': n_layer,
            'n_head': n_head,
            'n_embd': n_embd,
            'block_size': 3 * 1024, # Large enough for long sequences
            'dropout': dropout
        }

        self.transformer = nn.Sequential(*[Block(self.config) for _ in range(n_layer)]) #* unpacks the list
        
        self.embed_timestep = nn.Embedding(max_timestep, n_embd) #Embedding acts like a lookup table, if we used Linear, math would scale Step 100 100x larger than Step 1
        self.embed_state = nn.Linear(state_dim, n_embd)
        self.embed_action = nn.Linear(action_dim, n_embd)
        self.embed_reward = nn.Linear(1, n_embd)
        self.embed_ln = nn.LayerNorm(n_embd)
        
        # Predict action
        self.predict_action = nn.Sequential(
            nn.Linear(n_embd, action_dim),
            nn.Tanh()
        )

    def forward(self, states, actions, rewards, timesteps):
        # states: (batch, seq_len, state_dim)
        # actions: (batch, seq_len, action_dim)
        # rewards: (batch, seq_len, 1)
        # timesteps: (batch, seq_len)
        
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        time_embeddings = self.embed_timestep(timesteps)
        
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        reward_embeddings = self.embed_reward(rewards) + time_embeddings
        
        # Interleave: s_1, a_1, r_1, s_2, a_2, r_2...
        # We want to predict a_t given s_1, a_1, r_1, ..., s_t
        stacked_inputs = torch.stack(
            (state_embeddings, action_embeddings, reward_embeddings), dim=2
        ).reshape(batch_size, 3 * seq_len, self.config['n_embd'])
        
        stacked_inputs = self.embed_ln(stacked_inputs)
        
        x = self.transformer(stacked_inputs)
        
        # Reshape back to pull out state-based predictions
        x = x.reshape(batch_size, seq_len, 3, self.config['n_embd']).permute(0, 2, 1, 3) #We basically reshape it to have 3 folders: First of states, second of actions, and third of rewards
        
        # We want to predict action from state
        # The state embedding is at index 0 in the 3rd dimension
        action_preds = self.predict_action(x[:, 0]) #Taking the states folder
        
        return action_preds

    def get_action(self, states, actions, rewards, timesteps):
        # Used for inference
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.action_dim)
        rewards = rewards.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)
        
        preds = self.forward(states, actions, rewards, timesteps)
        return preds[0, -1] # Return last predicted action
