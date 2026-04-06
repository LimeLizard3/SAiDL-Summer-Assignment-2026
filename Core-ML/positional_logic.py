import torch
import torch.nn as nn
import math

def apply_rotary_emb(x, cos, sin):
    """
    x: (batch, n_heads, seq_len, d_k)
    cos/sin: (seq_len, d_k)
    """
    # 1. d_k is our "Head Dimension" (e.g. 32). 
    d_k = x.size(-1)
    
    # 2. Split the list into even and odd indices (creating coordinate pairs)
    x_even = x[..., 0::2] # Decimals 0, 2, 4...
    x_odd = x[..., 1::2]  # Decimals 1, 3, 5...
    
    # 3. Align cos and sin for broadcasting: (1, 1, seq_len, d_k/2)
    # We only take as many as we need for the current sequence.
    cos = cos[:x.size(2), :d_k//2].view(1, 1, x.size(2), d_k//2)
    sin = sin[:x.size(2), :d_k//2].view(1, 1, x.size(2), d_k//2)
    
    # 4. Standard Rotation Matrix applied to each pair:
    # out_x = x*cos - y*sin
    # out_y = x*sin + y*cos
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out

def get_alibi_slope(n_heads):
    """
    Standard ALiBi slope calculation.
    """
    def get_slopes_power_of_2(n):
        # The paper uses start = 2^(-8/n). 
        # Example for 4 heads: 2^(-2) = 0.25
        start = (2 ** (-8 / n))
        ratio = start
        # Returns [start^1, start^2, start^3, ...]
        return [start * (ratio**i) for i in range(n)]

    # If heads is a power of 2 (2, 4, 8, 16...), it's easy.
    if math.log2(n_heads).is_integer():
        return get_slopes_power_of_2(n_heads)
    else:
        # If not (like 6 heads), we interpolate from the next power of 2.
        # This ensures the slopes remain mathematically diverse.
        closest_power_of_2 = 2**math.floor(math.log2(n_heads))
        slopes_base = get_slopes_power_of_2(closest_power_of_2)
        slopes_extra = get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:n_heads - closest_power_of_2]
        return slopes_base + slopes_extra

def build_alibi_bias(n_heads, seq_len, device):
    """
    Returns a bias matrix of shape (n_heads, seq_len, seq_len)
    """
    # 1. Generate the slopes for each head
    slopes = torch.tensor(get_alibi_slope(n_heads), device=device).view(n_heads, 1, 1)
    
    # 2. Create the "Ruler" of indices: [0, 1, 2, 3...]
    m = torch.arange(seq_len, device=device)
    
    # 3. Create a 2D grid of distances by subtracting the ruler from itself.
    # Result: Grid[i, j] = j - i
    distance_matrix = (m.unsqueeze(0) - m.unsqueeze(1))
    
    # 4. Multiply slopes by distance to get the final penalty bias.
    # Words in the past (j < i) will get negative penalties.
    bias = slopes * distance_matrix 
    return bias.unsqueeze(0) # (1, n_heads, seq_len, seq_len)
