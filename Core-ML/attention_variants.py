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


class AFTAttention(nn.Module):
    """
    [Bonus Task] Attention-Free Transformer (AFT).
    Replaces dot-product attention with query-gated element-wise aggregation.
    Supports: simple, full, local, conv.
    """
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 1024, mode: str = "full", window_size: int = 50, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.mode = mode
        self.window_size = window_size
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # AFT positional weights w_{t,i}
        if mode == "full":
            self.wb = nn.Parameter(torch.randn(max_seq_len, max_seq_len) / math.sqrt(d_model))
        elif mode == "local" or mode == "conv":
            self.wb = nn.Parameter(torch.randn(max_seq_len) / math.sqrt(d_model))
        #.Parameter highlights the random numbers we generate so that backprop knows to tweak them
        #wb is called "weight bias"
        #Full mode tells AI to learn completely unique relationships for every possible pair of frames (that's why 2D)
        #Local/Conv mode tells AI to only care about relative distance and not exact pairs

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, pos_params=None):
        bsz, seq_len, _ = x.size()
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Q Gating (Sigmoid)
        q = torch.sigmoid(q) #This builds the knobs we were talking about RE Query Gating, and sigmoid is nice as it's all now b/w 0&1

        if self.mode == "simple":
            # Y_t = q_t * (sum(exp(k_i) * v_i) / sum(exp(k_i)))
            # Causal implementation using cumulative sums
            k_exp = torch.exp(k)
            kv = k_exp * v
            num = torch.cumsum(kv, dim=1) #Cumulative Sum
            den = torch.cumsum(k_exp, dim=1) + 1e-6 #dim = 1 is for seq_len, dim = 0 is for bsz
            #We use dim=1 as it represents TIME: The chronological list of frames
            #If we used dim-0, it's like adding frame 1 of game A to frame 1 of game B. It represents the No. of separate, II games
            output = q * (num / den)

        #Issue with simple is that it's not relative to time. A frame of 80% importance will have the same % 1000 frames later
            
        else:
            # Full, Local, and Conv modes involve a weight matrix W
            if self.mode == "full":
                w = self.wb[:seq_len, :seq_len]
            else: # local or conv
                # Relative weights: w[t, i] = wb[t-relative_dist]
                rel_pos = torch.arange(seq_len, device=x.device).unsqueeze(0) - torch.arange(seq_len, device=x.device).unsqueeze(1)
                #PyTorch knows that the dimensions don't match (Horizontal-Vertical), so it BROADCASTS and stretches the vertical & horizontal
                #ruler and subtracts the cells one by one and gets the relative position of each pair
                # rel_pos contains distances. We only care about causal side (dist >= 0)
                # wb index 0 is dist=0, index 10 is dist=10 behind.
                w = self.wb[torch.abs(rel_pos)] # Simple relative index (Takes res_pos) and outputs the corresponding pair
            
            # Apply causal masking to the weight matrix
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            
            # If mode is 'local', we also apply a sliding window mask
            if self.mode == "local" and self.window_size > 0: #Focuses on immediate local neighbourhood
                past_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device), diagonal=-self.window_size)
                causal_mask = causal_mask - past_mask
            
            # Combining w and k
            # score_{t,i} = w_{t,i} + k_i
            # We broadcast k over the target dimension t
            k_expanded = k.unsqueeze(1) # (Batch, 1, Time, Dimensions/Features)
            w_expanded = w.unsqueeze(0).unsqueeze(-1) # (1, Target time, Source time, 1)
            
            scores = w_expanded + k_expanded # (B, T, T, D) Broadcasting to the rescue yet again
            
            # Mask out future positions for causality
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(-1) == 0, float('-inf'))
            
            # Offset by max for stability
            max_scores = torch.max(scores, dim=2, keepdim=True)[0] #torch.max returnes values and indicies tuple, we only want values [0]
            #If we used Target Time (dim=1) instead, we would scan across target frames, instead finding the max score for a single past memory
            #across current frames (it would add vertically instead of horizontally, combining multiple frames when we're supposed to be looking at the target frame)
            #dim=2 basically says "look at the target row, and sum up all the prev. data entries in that row that came before it"

            exp_scores = torch.exp(scores - max_scores)
            #We do this for both num and den, meaning that the subtraction roughly cancels out. We do this to reduce the risk of a crash
            
            num = torch.sum(exp_scores * v.unsqueeze(1), dim=2) #v: [B,Target time, Source time, F] (after broadcasting)
            #Adding up the source time and squashing all the info from prev. frames and stuff it into target time
            den = torch.sum(exp_scores, dim=2) + 1e-6
            
            output = q * (num / den)
            
        return self.out_proj(output)
