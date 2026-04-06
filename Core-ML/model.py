import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from attention_variants import SlidingWindowAttention, MultiQueryAttention, LinearAttention
from positional_logic import apply_rotary_emb, build_alibi_bias
from conv_logic import CausalConv1d, DepthwiseSeparableCausalConv1d

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 1024):
        super().__init__()
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        pos = self.pe[:seq_len].transpose(0, 1) # (1, seq_len, d_model)
        return x + pos

class StandardAttention(nn.Module):
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
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, pos_params=None):
        bsz, seq_len, _ = x.size()
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
            scores = scores + pos_params["alibi_bias"][:, :, :seq_len, :seq_len] #This is adding the penalty but check the working of it again
            
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(output)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Select the attention engine based on config
        if config.attention_type == "mqa":
            self.attn = MultiQueryAttention(config.d_model, config.n_heads, config.dropout)
        elif config.attention_type == "linear":
            self.attn = LinearAttention(config.d_model, config.n_heads, config.dropout)
        elif config.attention_type == "sliding_window":
            self.attn = SlidingWindowAttention(config.d_model, config.n_heads, config.window_size, config.dropout)
        else: # Default is "standard"
            self.attn = StandardAttention(config.d_model, config.n_heads, config.dropout)
            
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        
        # [Task 4] Design A: Pre-Attention Convolution
        self.use_pre_conv = config.use_conv and config.conv_type == "pre_attention"
        if self.use_pre_conv:
            self.pre_conv = CausalConv1d(config.d_model, config.d_model, config.conv_kernel_size)

    def forward(self, x, mask=None, pos_params=None):
        if self.use_pre_conv:
            # Conv1d expects (B, Channels, Length)
            x_conv = x.transpose(1, 2)
            x_conv = self.pre_conv(x_conv)
            x = x_conv.transpose(1, 2)
            
        x = x + self.attn(self.ln1(x), mask=mask, pos_params=pos_params)
        x = x + self.ffn(self.ln2(x))
        return x

class ConvolutionBlock(nn.Module):
    """
    [Task 4] Design B: Interleaved Convolutional Block.
    Replaces an Attention layer with a Depthwise Separable 1D Convolution.
    """
    def __init__(self, config):
        super().__init__()
        self.conv = DepthwiseSeparableCausalConv1d(config.d_model, config.conv_kernel_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

    def forward(self, x, mask=None, pos_params=None):
        # mask and pos_params are unused here but kept for interface compatibility
        x = x + self.conv(self.ln1(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = PositionalEncoding(config.d_model, config.max_seq_len)
        self.drop = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList()
        for i in range(config.n_layers):
            # Design B: Interleave every 2nd layer if enabled
            if config.use_conv and config.conv_type == "interleaved" and i % 2 == 1:
                self.blocks.append(ConvolutionBlock(config))
            else:
                self.blocks.append(TransformerBlock(config))
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.token_emb.weight = self.head.weight

    def _generate_causal_mask(self, seq_len, device):
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.view(1, 1, seq_len, seq_len)

    def _get_rope_embeddings(self, seq_len, device):
        # Cache for RoPE sinusoids to avoid re-calculation
        if not hasattr(self, 'rope_cache') or self.rope_cache['cos'].size(0) < seq_len:
            d_k = self.config.d_model // self.config.n_heads
            inv_freq = 1.0 / (10000**(torch.arange(0, d_k, 2).float().to(device) / d_k))
            t = torch.arange(max(seq_len, self.config.max_seq_len), device=device).type_as(inv_freq)
            freqs = torch.einsum('i,j->ij', t, inv_freq)
            self.rope_cache = {
                'cos': freqs.cos(),
                'sin': freqs.sin()
            }
        return self.rope_cache['cos'][:seq_len], self.rope_cache['sin'][:seq_len]

    def forward(self, x):
        bsz, seq_len = x.size()
        mask = None
        pos_params = {}
        
        # Linear attention handles causality internally, doesn't need external mask
        if self.config.attention_type != "linear":
            mask = self._generate_causal_mask(seq_len, x.device)
            
        # [Task 3] Handle Positional Logic
        if self.config.pos_type == "rope":
            cos, sin = self._get_rope_embeddings(seq_len, x.device)
            pos_params["cos"], pos_params["sin"] = cos, sin
        elif self.config.pos_type == "alibi":
            pos_params["alibi_bias"] = build_alibi_bias(self.config.n_heads, seq_len, x.device)
        
        x = self.token_emb(x)
        
        # Only use absolute embeddings if not using modern variants
        if self.config.pos_type == "absolute":
            x = self.pos_emb(x)
            
        x = self.drop(x)
        
        for block in self.blocks:
            x = block(x, mask=mask, pos_params=pos_params)
            
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

