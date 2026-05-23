import torch  # type: ignore
import torch.nn as nn  # type: ignore
import math

def apply_rotary_emb(x, cos, sin):
    """
    Applies Rotary Positional Embeddings (RoPE) to input tensor x.
    Inputs:
        x: Query/Key activation tensor of shape [batch, n_heads, seq_len, d_k]
        cos/sin: Pre-computed cosine/sine components of shape [max_seq_len, d_k]
    Output:
        Rotated tensor of shape [batch, n_heads, seq_len, d_k]
    """
    # Retrieve head dimension d_k (scalar integer)
    d_k = x.size(-1)
    
    # Split the last dimension into even and odd indices to form coordinate pairs:
    # x_even shape: [batch, n_heads, seq_len, d_k/2]
    x_even = x[..., 0::2]
    # x_odd shape: [batch, n_heads, seq_len, d_k/2]
    x_odd = x[..., 1::2]
    
    # Slice cos/sin up to seq_len and d_k/2, then reshape to [1, 1, seq_len, d_k/2] for broadcasting
    cos = cos[:x.size(2), :d_k//2].view(1, 1, x.size(2), d_k//2)
    sin = sin[:x.size(2), :d_k//2].view(1, 1, x.size(2), d_k//2)
    
    # Initialize empty tensor of the same shape and type as x
    out = torch.empty_like(x)
    # Rotate even dimensions: out_even = x_even * cos - x_odd * sin
    out[..., 0::2] = x_even * cos - x_odd * sin
    # Rotate odd dimensions: out_odd = x_even * sin + x_odd * cos
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out

def get_alibi_slope(n_heads):
    """
    Calculates the geometric progression of decay slopes for ALiBi.
    Input:
        n_heads: Number of attention heads (scalar integer)
    Output:
        List of floats containing slopes for each head (length: n_heads)
    """
    def get_slopes_power_of_2(n):
        # Calculate base ratio for geometric progression: start = 2^(-8/n)
        start = (2 ** (-8 / n))
        ratio = start
        # Generate geometric series: [start, start^2, start^3, ..., start^n]
        return [start * (ratio**i) for i in range(n)]

    # If n_heads is a power of 2, calculate slopes directly
    if math.log2(n_heads).is_integer():
        return get_slopes_power_of_2(n_heads)
    else:
        # For non-power-of-2 heads, calculate slopes for the closest power of 2
        # and interpolate the remaining slopes to maintain diversity
        closest_power_of_2 = 2**math.floor(math.log2(n_heads))
        slopes_base = get_slopes_power_of_2(closest_power_of_2)
        slopes_extra = get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:n_heads - closest_power_of_2]
        return slopes_base + slopes_extra

def build_alibi_bias(n_heads, seq_len, device):
    """
    Generates the ALiBi bias matrix to penalize long-distance token interactions.
    Inputs:
        n_heads: Number of attention heads (scalar integer)
        seq_len: Current context length (scalar integer)
        device: PyTorch device on which to allocate the tensor
    Output:
        Bias tensor of shape [1, n_heads, seq_len, seq_len]
    """
    # Create slopes tensor and reshape to [n_heads, 1, 1] for 2D distance broadcasting
    slopes = torch.tensor(get_alibi_slope(n_heads), device=device).view(n_heads, 1, 1)
    
    # Create a range of indices from 0 to seq_len-1 (shape [seq_len])
    m = torch.arange(seq_len, device=device)
    
    # Compute relative distance matrix using broadcasting subtraction:
    # m.unsqueeze(0) shape: [1, seq_len]
    # m.unsqueeze(1) shape: [seq_len, 1]
    # distance_matrix shape: [seq_len, seq_len], where entry (i, j) is j - i
    distance_matrix = (m.unsqueeze(0) - m.unsqueeze(1))
    
    # Apply slope scaling to distance matrix: shape [n_heads, seq_len, seq_len]
    # Future tokens (j > i) get positive values (will be masked anyway)
    # Past tokens (j < i) get negative values (distance penalty)
    bias = slopes * distance_matrix 
    return bias.unsqueeze(0) # Reshape to [1, n_heads, seq_len, seq_len]
