import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class RunningMeanStd:
    """Tracks the running mean and standard deviation of data."""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon #Btw, we use float64 instead of float32 because the count will get so large that adding a tiny new value has virtually no effect. float64 prevents
                             #catastrophic precision loss

    def update(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0) #axis = 0 means calculate on each independent COLUMN rather than row IE it calculates mean and var over each feature/column
        batch_count = x.shape[0] #1st dimension gives batchsize 
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count): #This is a pure funciton; when combining different models, keeping this outside the class
    delta = batch_mean - mean                                                                 #just makes more sense as it wont update any other vars. Good for testing and portability
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count #Updates running mean by shifting old mean towards new mean scaled by how large batch is relative to total history
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count #Calculating a BRAND NEW OVERALL MEAN. ma & mb are still centered around old mean. ManvsChild problem
    new_var = M2 / tot_count #Var is an average, so we divide by tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class Normalizer:
    """Applies running mean/std normalization to states/observations."""
    def __init__(self, shape, clip_limit=10):
        self.rms = RunningMeanStd(shape=shape)
        self.clip_limit = clip_limit

    def __call__(self, x, update=True): #Thanks to call, we can treat normalizer = Normalizer(...) like a function instead of having to write normalizer.name(...)
        if update:
            self.rms.update(x) #If update = True, passes data to Chan's algo to calculate RMS and Var
        x = (x - self.rms.mean) / (np.sqrt(self.rms.var) + 1e-8) #Z-Score Standardization. Numerator centers data around 0 and denominator ensures that X & Y axis are on the same scale
        x = np.clip(x, -self.clip_limit, self.clip_limit) #Ensuring things don't go too out of hand despite normalization; This is a safety precaution
        return x

    def save(self, filename):
        """Save the normalizer stats to a file."""
        np.savez(filename + "_normalizer.npz", mean=self.rms.mean, var=self.rms.var, count=self.rms.count)

    def load(self, filename):
        """Load the normalizer stats from a file."""
        data = np.load(filename + "_normalizer.npz")
        self.rms.mean = data["mean"]
        self.rms.var = data["var"]
        self.rms.count = data["count"]

class Actor(nn.Module):
    """The Actor policy mapping states to actions."""
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action
        #If we put these layers inside forward(), every single time the agent took a step, w and b would be re-calculated again and again. Instead, we put them in __init__
        #so that they're calculated once along with the rest of the brain

    def forward(self, state): #forward is a reserved word. If you call Actor(data) it'll automatically run the data past forward, unlike for other functions
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a)) #mult can output any No. in the universe, so if we wrap it around tanh, we're basically squashing it into a fixed
                                                        #guaranteed percentage range. That way, the physics engine doesn't glitch out
        #IMPORTANT: Why not LeakyReLU??? First, ReLU is computationally cheap. Second, sometimes a neuron needs to shut up (O=nothin here), and Third, this network is incredibly small with
        #only two layers. There's really no need for it here. However, if the agent flatlines and refuses to learn, switching to LeakyReLU is one of the first debugging steps


class Critic(nn.Module):
    """The Critic network predicting Q-values for (state, action) pairs."""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__() #Calls the setup of the parent class nn.Module before it setsup
        # Q1 network architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 network architecture (The 'Twin' in TD3)
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim) #Needs to see both environment state and actor's action choice to see if it's a good pairing
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1) #1st layer expects both of these, so, we concatenate them together. 1 means along the column (Batches remain intact)

        # Forward pass for Q1
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1) #ReLU will break our negative scores (Even LeakyReLU will), so we don't run it through ReLU again

        # Forward pass for Q2
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        """Utility for calculating only the first Q-value."""
        sa = torch.cat([state, action], 1) #With axis = 1, each row contains the full picture actions and state together!
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

#We now move to an agent that can actively remember the past

class TransformerActor(nn.Module):
    """
    A Causal Transformer Actor that uses history to decide actions.
    Adapted from the Core-ML Advanced Transformer.
    """
    def __init__(self, state_dim, action_dim, max_action, d_model=128, n_heads=4, n_layers=2, dropout=0.1):
        super(TransformerActor, self).__init__()
        self.max_action = max_action
        
        # 1. Input Embedding: Maps raw sensors to Transformer space
        self.state_emb = nn.Linear(state_dim, d_model)
        self.action_emb = nn.Linear(action_dim, d_model)
        self.drop = nn.Dropout(dropout)
        #Transformer requires all inputs to be exactly d_model, we're just projecting data into the right dimension
        # 2. Transformer Blocks
        self.blocks = nn.ModuleList([ #ModuleList uses Py list comprehension to create N_layers identical transformerblocks
                                      #& stack them into a list. This is where all the attention math happens
            TransformerBlock_RL(d_model, n_heads, dropout) for _ in range(n_layers) #Passing it into the basic Transformer
        ])
        #Note, what's passed here aren't required parameters, they're construction tools

        # 3. LayerNorm and Final Head
        self.ln_f = nn.LayerNorm(d_model) #Final LayerNorm before the action head to keep things stable
        self.head = nn.Linear(d_model, action_dim) #O/P layer

    @property #Allows us to define a funcn in a class, but access it as though it was a simple var.
    #my_model.device instead of my_model.device(self). Acts like a "read-only" property
    def device(self):
        """Infers device from model parameters."""
        return next(self.parameters()).device #next() grabs the very first weight matrix

    def select_action(self, state_norm, state_history, action_history, return_attn=False):
        """Allows for single-step inference while maintaining temporal context."""
        state_seq = torch.FloatTensor(np.array(state_history)).unsqueeze(0).to(self.device)
        action_seq = torch.FloatTensor(np.array(action_history)).unsqueeze(0).to(self.device)
        
        if return_attn:
            action, attn = self.forward(state_seq, action_seq, return_attn=True)
            return action.cpu().data.numpy().flatten(), attn.cpu().data.numpy()
        
        return self.forward(state_seq, action_seq).cpu().data.numpy().flatten()

    def forward(self, state_seq, action_seq, return_attn=False): #Remember, state_seq handles the 'where I am' and action_seq handles the 'what I did'
        # state_seq: (Batch size, Sequence length, state_dim)
        # action_seq: (B, L, action_dim)
        #Same goes for x fyi (x[2] is d_model = 128)
        
        # Sum the state and action embeddings (standard for Decision Transformers)
        x = self.state_emb(state_seq) + self.action_emb(action_seq) #Projecting and combining
        x = self.drop(x) #"Blinding"
        
        # Build a causal mask (don't look at the future!) - This is the core 'Time Travel Prevention'
        seq_len = x.size(1) 
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
        #torch.ones creates a seq_len square grid on device (cuda)
        #torch.tril turns the top right half of that grid into 0s (it's like drawing a diagonal). 0 acts like a brick wall
        #preventing future vision (0 at step 3 prevents looking at steps 4,5, or 6)
        #view(1, 1, seq_len, seq_len) reshapes it to (1, 1, seq_len, seq_len) to match the dimensions of the attention scores
        #this is done for BROADCASTING

        all_attns = []
        for block in self.blocks:
            if return_attn:
                x, attn = block(x, mask=mask, return_attn=True)
                all_attns.append(attn)
            else:
                x = block(x, mask=mask)
            
        x = self.ln_f(x)
        
        # We only care about the very last decision in the sequence - predicting the NEXT action
        last_token = x[:, -1, :]
        action = self.max_action * torch.tanh(self.head(last_token))
        
        if return_attn:
            # Return action and average attention across blocks or just the last block
            return action, torch.stack(all_attns).mean(dim=0) # Shape: (B, H, L, L)
            
        return action


class TransformerBlock_RL(nn.Module):
    """Simplified Transformer Block for RL - Adapted from Main Assignment."""
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.attn = StandardAttention_RL(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), #Standard 4x expansion in the FeedForward network (Scratchpad)
            nn.GELU(), #Using GELU as it helps with the vanishing gradient problem better than RELU here
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, return_attn=False):
        # Using Pre-LayerNorm architecture as it is generally more stable for RL agents
        attn_out, attn_weights = self.attn(self.ln1(x), mask=mask)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        if return_attn:
            return x, attn_weights
        return x


class StandardAttention_RL(nn.Module):
    """Standard Causal Attention adapted for RL sequences."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads #Ensuring dimensionality matches across heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model) #After the n_heads think, this blends them all together
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        bsz, seq_len, _ = x.size()
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention score calculation with scaling factor to prevent large values
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf')) #Applying the 'Time Travel' mask. masked_fill finally puts on the mask after being passed again and again
            #Scores dimensions: (bsz, n_heads, seq_len, seq_len) where first seq_len is for q and the other is for k
            #The mask is a lower triangular matrix (only 1s on and below the diagonal)
            
        attn_weights = F.softmax(scores, dim=-1) #dim=-1 tells Softmax to go to the last dimension (Keys)
        output = torch.matmul(self.dropout(attn_weights), v)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.out_proj(output), attn_weights
