import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
import math
from positional_logic import apply_rotary_emb

class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention.
    Restricts the receptive field to a local context window to reduce memory bandwidth.
    """
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
        # Input shape x: [batch_size, seq_len, d_model]
        bsz, seq_len, _ = x.size()
        
        # Projections to Q, K, V; reshaped to: [batch_size, n_heads, seq_len, d_k]
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply RoPE positional rotations if provided
        if pos_params is not None and "cos" in pos_params:
            q = apply_rotary_emb(q, pos_params["cos"], pos_params["sin"])
            k = apply_rotary_emb(k, pos_params["cos"], pos_params["sin"])
        
        # Calculate raw dot-product attention scores; shape: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply ALiBi bias penalty if provided
        if pos_params is not None and "alibi_bias" in pos_params:
            scores = scores + pos_params["alibi_bias"][:, :, :seq_len, :seq_len]
        
        # Create sliding window causal mask: shape [seq_len, seq_len]
        device = x.device
        window_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        
        # Mask out tokens that are further in the past than window_size
        if self.window_size > 0:
            past_mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=-self.window_size)
            window_mask = window_mask - past_mask
        
        # Reshape for broadcasting over batch and head dimensions: [1, 1, seq_len, seq_len]
        window_mask = window_mask.view(1, 1, seq_len, seq_len)
        
        # Mask out unavailable tokens by filling with negative infinity
        scores = scores.masked_fill(window_mask == 0, float('-inf'))
            
        # Standard softmax activation
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Multiply attention weights by values; output shape: [batch_size, n_heads, seq_len, d_k]
        output = torch.matmul(attn_weights, v)
        # Reshape back to sequence representation shape: [batch_size, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(output)

class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA).
    Shares a single Key and Value head across all Query heads to reduce memory bandwidth.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Q receives full projection dimension, while K & V receive single-head dimension
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.d_k) 
        self.v_proj = nn.Linear(d_model, self.d_k)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, pos_params=None):
        # Input shape x: [batch_size, seq_len, d_model]
        bsz, seq_len, _ = x.size()
        
        # Q shape: [batch_size, n_heads, seq_len, d_k]
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # K and V shape: [batch_size, 1, seq_len, d_k] (only a single head)
        k = self.k_proj(x).view(bsz, seq_len, 1, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, 1, self.d_k).transpose(1, 2)
        
        # Apply RoPE positional rotations if provided
        if pos_params is not None and "cos" in pos_params:
            q = apply_rotary_emb(q, pos_params["cos"], pos_params["sin"])
            k = apply_rotary_emb(k, pos_params["cos"], pos_params["sin"])
            
        # Compute attention scores; Single K head is broadcasted over all Q heads:
        # scores shape: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply ALiBi bias penalty if provided
        if pos_params is not None and "alibi_bias" in pos_params:
            scores = scores + pos_params["alibi_bias"][:, :, :seq_len, :seq_len]
        
        # Apply causal masking
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # Standard softmax activation
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Multiply attention weights by values (V is broadcasted to match heads);
        # output shape: [batch_size, n_heads, seq_len, d_k]
        output = torch.matmul(attn_weights, v)
        # Reshape to final representation shape: [batch_size, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(output)

class LinearAttention(nn.Module):
    """
    Causal Linear Attention.
    Re-orders matrix multiplication from (Q*K)*V to Q*(K*V) to bypass the O(N^2) similarity matrix.
    Uses cumulative sums over time to enforce causality.
    """
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
        # Input shape x: [batch_size, seq_len, d_model]
        bsz, seq_len, _ = x.size()
        
        # Standard projections to Q, K, V; shape: [batch_size, n_heads, seq_len, d_k]
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply RoPE positional rotations (Linear Attention depends on RoPE for positional stability)
        if pos_params is not None and "cos" in pos_params:
            q = apply_rotary_emb(q, pos_params["cos"], pos_params["sin"])
            k = apply_rotary_emb(k, pos_params["cos"], pos_params["sin"])
        
        # Apply a ReLU-based kernel activation to ensure non-negative values in denominator
        q = F.relu(q) + 1e-6
        k = F.relu(k) + 1e-6
        
        # Compute outer product of Key and Value vectors for each timestep:
        # kv shape: [batch_size, n_heads, seq_len, d_k, d_k]
        kv = torch.einsum('bhtd,bhte->bhtde', k, v)
        # Cumulative sum along the sequence dimension (dim=2) to enforce causality
        # kv_cumsum shape: [batch_size, n_heads, seq_len, d_k, d_k]
        kv_cumsum = torch.cumsum(kv, dim=2)
        
        # Compute numerator output: project Q onto the causal KV memory accumulation
        # num shape: [batch_size, n_heads, seq_len, d_k]
        num = torch.einsum('bhtd,bhtde->bhte', q, kv_cumsum)
        
        # Compute normalizer denominator to stabilize scaling over sequence length
        # k_cumsum shape: [batch_size, n_heads, seq_len, d_k]
        k_cumsum = torch.cumsum(k, dim=2)
        # den shape: [batch_size, n_heads, seq_len, 1]
        den = torch.einsum('bhtd,bhtd->bht', q, k_cumsum).unsqueeze(-1) + 1e-6
        
        # Normalize: shape [batch_size, n_heads, seq_len, d_k]
        output = num / den
        
        # Reshape to final sequence representation shape: [batch_size, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(output)

class AFTAttention(nn.Module):
    """
    Attention-Free Transformer (AFT).
    Bypasses dot-product attention entirely using a query-gated, element-wise aggregation.
    Supports modes: simple, full, local, conv.
    """
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 1024, mode: str = "full", window_size: int = 50, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.mode = mode
        self.window_size = window_size
        
        # Linear projections for Queries, Keys, and Values
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Instantiate learned positional weight parameters
        if mode == "full":
            # Learn a unique pairwise bias for every position pair: shape [max_seq_len, max_seq_len]
            self.wb = nn.Parameter(torch.randn(max_seq_len, max_seq_len) / math.sqrt(d_model))
        elif mode == "local" or mode == "conv":
            # Learn relative distance-based biases: shape [max_seq_len]
            self.wb = nn.Parameter(torch.randn(max_seq_len) / math.sqrt(d_model))

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, pos_params=None):
        # Input shape x: [batch_size, seq_len, d_model]
        bsz, seq_len, _ = x.size()
        
        # Project inputs; output shapes: [batch_size, seq_len, d_model]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Apply Query Gating activation (sigmoid constraint between 0 and 1)
        q = torch.sigmoid(q)

        if self.mode == "simple":
            # AFT-Simple variant:
            # y_t = q_t * (sum_{i<=t}(exp(k_i)*v_i) / sum_{i<=t}(exp(k_i)))
            k_exp = torch.exp(k)
            kv = k_exp * v
            # Cumulative sums along the sequence length dimension (dim=1) to ensure causality
            num = torch.cumsum(kv, dim=1)
            den = torch.cumsum(k_exp, dim=1) + 1e-6
            output = q * (num / den)
            
        else:
            # Extract and shape the relevant positional weight matrix W: [seq_len, seq_len]
            if self.mode == "full":
                w = self.wb[:seq_len, :seq_len]
            else:
                # Compute 2D relative index distances grid via broadcasting subtraction: shape [seq_len, seq_len]
                rel_pos = torch.arange(seq_len, device=x.device).unsqueeze(0) - torch.arange(seq_len, device=x.device).unsqueeze(1)
                # Map distances to corresponding parameter values using absolute index lookup
                w = self.wb[torch.abs(rel_pos)]
            
            # Create base causal mask
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            
            # Apply local sliding window restriction if mode is local
            if self.mode == "local" and self.window_size > 0:
                past_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device), diagonal=-self.window_size)
                causal_mask = causal_mask - past_mask
            
            # Broadcast Key and Weight tensors together to compute scores:
            # k_expanded shape: [batch_size, 1, seq_len, d_model] (target query time dimension inserted)
            k_expanded = k.unsqueeze(1)
            # w_expanded shape: [1, seq_len, seq_len, 1] (batch and feature dimensions inserted)
            w_expanded = w.unsqueeze(0).unsqueeze(-1)
            
            # Add together: scores shape: [batch_size, seq_len, seq_len, d_model]
            scores = w_expanded + k_expanded
            
            # Apply causal mask: mask shape [1, seq_len, seq_len, 1]
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(-1) == 0, float('-inf'))
            
            # Extract maximum score along the source time dimension (dim=2) for numerical stability
            max_scores = torch.max(scores, dim=2, keepdim=True)[0]
            # Exponentiate stable scores
            exp_scores = torch.exp(scores - max_scores)
            
            # Weighted sum over values along the source time dimension: output shape [batch_size, seq_len, d_model]
            num = torch.sum(exp_scores * v.unsqueeze(1), dim=2)
            # Sum of exponential weights: output shape [batch_size, seq_len, d_model]
            den = torch.sum(exp_scores, dim=2) + 1e-6
            
            # Multiply query-gated activations by attention values
            output = q * (num / den)
            
        return self.out_proj(output)
