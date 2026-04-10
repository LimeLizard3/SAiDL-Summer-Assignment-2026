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
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count): #This is a pure funciton; when combining different models, keeping this outside the class
    delta = batch_mean - mean                                                                 #just makes more sense as it wont update any other vars. However, ask once.
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class Normalizer:
    """Applies running mean/std normalization to states/observations."""
    def __init__(self, shape, clip_limit=10):
        self.rms = RunningMeanStd(shape=shape)
        self.clip_limit = clip_limit

    def __call__(self, x, update=True):
        if update:
            self.rms.update(x)
        x = (x - self.rms.mean) / (np.sqrt(self.rms.var) + 1e-8)
        x = np.clip(x, -self.clip_limit, self.clip_limit) #Ensuring things don't go too out of hand despite normalization; This is a safety precaution
        return x

class Actor(nn.Module):
    """The Actor policy mapping states to actions."""
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state): #forward is a reserved word. If you call Actor(data) it'll automatically run the data past forward, unlike for other functions
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    """The Critic network predicting Q-values for (state, action) pairs."""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # Q1 network architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 network architecture (The 'Twin' in TD3)
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        # Forward pass for Q1
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

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
        
        # 2. Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock_RL(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        
        # 3. LayerNorm and Final Head
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, action_dim)

    def forward(self, state_seq, action_seq):
        # state_seq: (B, L, state_dim)
        # action_seq: (B, L, action_dim)
        
        # Sum the state and action embeddings (standard for Decision Transformers)
        x = self.state_emb(state_seq) + self.action_emb(action_seq)
        x = self.drop(x)
        
        # Build a causal mask (don't look at the future!)
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
        
        for block in self.blocks:
            x = block(x, mask=mask)
            
        x = self.ln_f(x)
        
        # We only care about the very last decision in the sequence
        last_token = x[:, -1, :]
        return self.max_action * torch.tanh(self.head(last_token))


class TransformerBlock_RL(nn.Module):
    """Simplified Transformer Block for RL."""
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.attn = StandardAttention_RL(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.ffn(self.ln2(x))
        return x


class StandardAttention_RL(nn.Module):
    """Standard Causal Attention adapted for RL sequences."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        bsz, seq_len, _ = x.size()
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(self.dropout(attn_weights), v)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.out_proj(output)
