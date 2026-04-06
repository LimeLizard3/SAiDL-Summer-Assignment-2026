import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from positional_logic import apply_rotary_emb

class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, window_size: int = 50, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, pos_params=None):
        bsz, seq_len, _ = x.size()
        
        # Standard projection
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # [Task 3] Apply RoPE if provided
        if pos_params is not None and "cos" in pos_params:
            q = apply_rotary_emb(q, pos_params["cos"], pos_params["sin"])
            k = apply_rotary_emb(k, pos_params["cos"], pos_params["sin"])
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # [Task 3] Apply ALiBi bias if provided
        if pos_params is not None and "alibi_bias" in pos_params:
            scores = scores + pos_params["alibi_bias"][:, :, :seq_len, :seq_len]
        
        # Generate the normal causal mask ensuring we can't look into the future
        device = x.device
        window_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        
        # Magic of Sliding Window: We literally subtract the 1s that are too far in the past!
        # This forces the matrix to only have a small diagonal band of "1"s (the window).
        if self.window_size > 0:
            past_mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=-self.window_size)
            window_mask = window_mask - past_mask
        
        window_mask = window_mask.view(1, 1, seq_len, seq_len)
        
        # Overwrite the standard mask logic
        scores = scores.masked_fill(window_mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(output)


class MultiQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # MAGIC: Q gets full projection...
        self.q_proj = nn.Linear(d_model, d_model)
        # ...but Keys and Values ONLY get 1 single head of projection! 
        # This saves massive VRAM.
        self.k_proj = nn.Linear(d_model, self.d_k) 
        self.v_proj = nn.Linear(d_model, self.d_k)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, pos_params=None):
        bsz, seq_len, _ = x.size()
        
        # Q shape: (Batch, Heads, Length, Dim)
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # K, V shape: (Batch, 1, Length, Dim)
        k = self.k_proj(x).view(bsz, seq_len, 1, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, 1, self.d_k).transpose(1, 2)
        
        # [Task 3] Apply RoPE if provided
        if pos_params is not None and "cos" in pos_params:
            q = apply_rotary_emb(q, pos_params["cos"], pos_params["sin"])
            k = apply_rotary_emb(k, pos_params["cos"], pos_params["sin"])
            
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # [Task 3] Apply ALiBi bias if provided
        if pos_params is not None and "alibi_bias" in pos_params:
            scores = scores + pos_params["alibi_bias"][:, :, :seq_len, :seq_len]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(output)


class LinearAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None, pos_params=None):
        bsz, seq_len, _ = x.size()
        
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # [Task 3] Apply RoPE if provided (Linear Attention only supports RoPE)
        if pos_params is not None and "cos" in pos_params:
            q = apply_rotary_emb(q, pos_params["cos"], pos_params["sin"])
            k = apply_rotary_emb(k, pos_params["cos"], pos_params["sin"])
        
        # In Linear Attention, we delete the Softmax operation entirely!
        # Instead, we just force Q and K to be strictly positive numbers using ReLU.
        q = F.relu(q) + 1e-6
        k = F.relu(k) + 1e-6
        
        # EXPLANATION OF THE MATH TRICK:
        # Standard: Output = (Q * K) * V   <- The Q*K step creates a massive (Seq, Seq) matrix.
        # Linear:   Output = Q * (K * V)   <- Math says we can re-order this! 
        # Below, we are literally multiplying K and V first, and then applying Q. 
        # This completely skips building the N x N grid!
        
        # To make it causal (so word 10 can't read word 11's memory), 
        # we calculate a running combined sum of matrix (K * V) over time.
        kv = torch.einsum('bhtd,bhte->bhtde', k, v)
        kv_cumsum = torch.cumsum(kv, dim=2)
        
        # Apply Q strictly to the moving sum of memory.
        num = torch.einsum('bhtd,bhtde->bhte', q, kv_cumsum)
        
        # Normalize the result so the numbers don't explode to infinity
        k_cumsum = torch.cumsum(k, dim=2)
        den = torch.einsum('bhtd,bhtd->bht', q, k_cumsum).unsqueeze(-1) + 1e-6
        
        output = num / den
        
        # And we are done! Zero (Seq x Seq) massive matrixes were generated!
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(output)
