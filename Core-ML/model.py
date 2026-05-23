import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
import math
from attention_variants import SlidingWindowAttention, MultiQueryAttention, LinearAttention, AFTAttention
from positional_logic import apply_rotary_emb, build_alibi_bias
from conv_logic import CausalConv1d, DepthwiseSeparableCausalConv1d

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal absolute positional encoding module.
    Adds position-dependent values to token embedding representations.
    Pe shape initialized as: [max_seq_len, 1, d_model]
    """
    def __init__(self, d_model: int, max_seq_len: int = 1024):
        super().__init__()
        # Generate position index column tensor: shape [max_seq_len, 1]
        position = torch.arange(max_seq_len).unsqueeze(1)
        # Compute divisor frequencies: shape [d_model / 2]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # Initialize pe tensor: shape [max_seq_len, 1, d_model]
        pe = torch.zeros(max_seq_len, 1, d_model)
        # Populate even dimensions with sine functions
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        # Populate odd dimensions with cosine functions
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # Register as a buffer so it is saved with the model but not trained
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Input tensor x shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        # Slice pe up to current sequence length and transpose to [1, seq_len, d_model] for batch broadcasting
        pos = self.pe[:seq_len].transpose(0, 1)
        return x + pos

class StandardAttention(nn.Module):
    """
    Standard Multi-Head Scaled Dot-Product Attention.
    Supports causal masking, Rotary Embeddings (RoPE), and ALiBi bias injection.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        # Dimension of each individual attention head (scalar integer)
        self.d_k = d_model // n_heads
        
        # Linear projection matrices for Queries, Keys, and Values
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        # Output projection matrix to mix head outputs
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, pos_params=None):
        # Input tensor x shape: [batch_size, seq_len, d_model]
        bsz, seq_len, _ = x.size()
        
        # Project inputs to Q, K, V and reshape to split heads:
        # Output shapes: [batch_size, n_heads, seq_len, d_k]
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply RoPE rotations to Q and K if parameters are passed in pos_params
        if pos_params is not None and "cos" in pos_params:
            q = apply_rotary_emb(q, pos_params["cos"], pos_params["sin"])
            k = apply_rotary_emb(k, pos_params["cos"], pos_params["sin"])
            
        # Compute raw attention score weights via batched matrix multiplication:
        # q shape: [batch_size, n_heads, seq_len, d_k]
        # k.transpose(-2, -1) shape: [batch_size, n_heads, d_k, seq_len]
        # scores shape: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add ALiBi relative position bias penalty to attention scores if passed in pos_params
        if pos_params is not None and "alibi_bias" in pos_params:
            scores = scores + pos_params["alibi_bias"][:, :, :seq_len, :seq_len]
            
        # Mask future positions with negative infinity to preserve autoregressive causality
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # Normalize scores to probabilities across the key sequence dimension
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum over values: output shape [batch_size, n_heads, seq_len, d_k]
        output = torch.matmul(attn_weights, v)
        # Transpose back and flatten heads: output shape [batch_size, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(output)

class TransformerBlock(nn.Module):
    """
    Standard self-attention Transformer block.
    Integrates Pre-Attention 1D Causal Convolution option.
    """
    def __init__(self, config):
        super().__init__()
        # Select target self-attention module configuration
        self.attn: nn.Module
        if config.attention_type == "mqa":
            self.attn = MultiQueryAttention(config.d_model, config.n_heads, config.dropout)
        elif config.attention_type == "linear":
            self.attn = LinearAttention(config.d_model, config.n_heads, config.dropout)
        elif config.attention_type == "sliding_window":
            self.attn = SlidingWindowAttention(config.d_model, config.n_heads, config.window_size, config.dropout)
        elif config.attention_type == "aft":
            self.attn = AFTAttention(config.d_model, config.n_heads, config.max_seq_len, config.aft_mode, config.aft_window_size, config.dropout)
        else:
            self.attn = StandardAttention(config.d_model, config.n_heads, config.dropout)
            
        # Standard Multi-Layer Perceptron (MLP) / Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        
        # Pre-Attention Convolution layer initialization
        self.use_pre_conv = config.use_conv and config.conv_type == "pre_attention"
        if self.use_pre_conv:
            self.pre_conv = CausalConv1d(config.d_model, config.d_model, config.conv_kernel_size)

    def forward(self, x, mask=None, pos_params=None):
        # Input tensor x shape: [batch_size, seq_len, d_model]
        if self.use_pre_conv:
            # Transpose to [batch_size, d_model, seq_len] for Conv1D filtering, then transpose back
            x_conv = x.transpose(1, 2)
            x_conv = self.pre_conv(x_conv)
            x = x_conv.transpose(1, 2)
            
        # Residual connections with LayerNorm
        x = x + self.attn(self.ln1(x), mask=mask, pos_params=pos_params)
        x = x + self.ffn(self.ln2(x))
        return x

class ConvolutionBlock(nn.Module):
    """
    Interleaved Depthwise-Separable 1D Causal Convolutional Block.
    Replaces global self-attention with efficient local kernel operations.
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
        # Input tensor x shape: [batch_size, seq_len, d_model]
        # Transpose input normalized state to [batch_size, d_model, seq_len] for convolution
        conv_out = self.conv(self.ln1(x).transpose(1, 2)).transpose(1, 2)
        x = x + conv_out
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    """
    The main modular Language Model class wrapper.
    Responsible for token embeddings, positional choices, caching, layer routing, and output prediction.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Input embedding matrix: maps token IDs to dense representations
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        # Sinusoidal absolute position encodings
        self.pos_emb = PositionalEncoding(config.d_model, config.max_seq_len)
        self.drop = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList()
        for i in range(config.n_layers):
            # Design B: Alternate attention blocks with convolution blocks if interleaved enabled
            if config.use_conv and config.conv_type == "interleaved" and i % 2 == 1:
                self.blocks.append(ConvolutionBlock(config))
            else:
                self.blocks.append(TransformerBlock(config))
        self.ln_f = nn.LayerNorm(config.d_model)
        # Final prediction head; outputs logits mapping back to vocabulary dimensions
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie input embeddings with output projection weights to save parameter allocation
        self.token_emb.weight = self.head.weight

    def _generate_causal_mask(self, seq_len, device):
        """
        Creates a lower-triangular causal attention mask of shape [1, 1, seq_len, seq_len].
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.view(1, 1, seq_len, seq_len)

    def _get_rope_embeddings(self, seq_len, device):
        """
        Computes or retrieves cached sine and cosine terms for RoPE frequency encoding.
        Outputs: (cos, sin) tensors of shape [seq_len, d_k]
        """
        if not hasattr(self, 'rope_cache') or self.rope_cache['cos'].size(0) < seq_len:
            d_k = self.config.d_model // self.config.n_heads
            # Calculate inverse frequencies for relative positioning
            inv_freq = 1.0 / (10000**(torch.arange(0, d_k, 2).float().to(device) / d_k))
            t = torch.arange(max(seq_len, self.config.max_seq_len), device=device).type_as(inv_freq)
            # Outer product of time index and frequencies: shape [seq_len, d_k/2]
            freqs = torch.einsum('i,j->ij', t, inv_freq)
            self.rope_cache = {
                'cos': freqs.cos(),
                'sin': freqs.sin()
            }
        return self.rope_cache['cos'][:seq_len], self.rope_cache['sin'][:seq_len]

    def forward(self, x):
        # Input shape x: [batch_size, seq_len] containing token integer IDs
        bsz, seq_len = x.size()
        mask = None
        pos_params = {}
        
        # Linear attention formats internal causal features natively, bypassing causal matrix multiplication masks
        if self.config.attention_type != "linear":
            mask = self._generate_causal_mask(seq_len, x.device)
            
        # Extract relevant positional parameters based on config settings
        if self.config.pos_type == "rope":
            cos, sin = self._get_rope_embeddings(seq_len, x.device)
            pos_params["cos"], pos_params["sin"] = cos, sin
        elif self.config.pos_type == "alibi":
            pos_params["alibi_bias"] = build_alibi_bias(self.config.n_heads, seq_len, x.device)
        
        # Map IDs to token embeddings; output shape: [batch_size, seq_len, d_model]
        x = self.token_emb(x)
        
        # Inject standard absolute positional encodings if selected
        if self.config.pos_type == "absolute":
            x = self.pos_emb(x)
            
        x = self.drop(x)
        
        # Run sequentially through each architectural block
        for block in self.blocks:
            x = block(x, mask=mask, pos_params=pos_params)
            
        # Final normalization step
        x = self.ln_f(x)
        # Project representation to output vocabulary logits: shape [batch_size, seq_len, vocab_size]
        logits = self.head(x)
        return logits
