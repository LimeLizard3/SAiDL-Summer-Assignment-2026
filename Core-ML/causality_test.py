import torch  # type: ignore
import torch.nn as nn  # type: ignore
from config import TransformerConfig
from model import TransformerLM  # type: ignore

def test_causality(attention_type="standard", aft_mode="full", use_conv=False):
    """
    Test script to verify that future tokens do not affect past predictions (causality/autoregressive property).
    """
    desc = f"{attention_type.upper()}"
    if attention_type == "aft":
        desc += f" ({aft_mode})"
    if use_conv:
        desc += " + CONV"
        
    print(f"\n--- Testing Causality for: {desc} ---")
    
    # Configure device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Instantiate transformer configurations
    config = TransformerConfig()
    config.attention_type = attention_type
    config.aft_mode = aft_mode
    config.use_conv = use_conv
    config.n_layers = 1 # A single layer is sufficient to test gradient/information flow leakage
    
    # Load model and place in evaluation mode
    model = TransformerLM(config).to(device)
    model.eval()
    
    # 1. Create a base sequence X of shape [1, seq_len] with random token integer IDs
    seq_len = 20
    x1 = torch.randint(0, config.vocab_size, (1, seq_len)).to(device)
    
    # 2. Clone sequence X to create sequence Y (x2)
    split_idx = 10
    x2 = x1.clone()
    # Modify ONLY a single token at split_idx.
    # If causal masking is correct, predictions at indices < split_idx must remain identical.
    x2[0, split_idx] = (x1[0, split_idx] + 7) % config.vocab_size
    
    # Forward pass without calculating gradients
    with torch.no_grad():
        # Output tensor shape: [1, seq_len, vocab_size]
        output1 = model(x1)
        output2 = model(x2)
        
    # 3. Measure maximum absolute difference between outputs before split_idx
    # Slices are taken up to split_idx along the sequence dimension: shape [1, split_idx, vocab_size]
    diff = torch.abs(output1[0, :split_idx] - output2[0, :split_idx]).max().item()
    
    # If the difference is below numerical threshold (1e-4), causality holds
    if diff < 1e-4:
        print(f"  PASS: No leakage. Max diff: {diff:.2e}")
    else:
        print(f"  FAILED: Leakage detected. Max diff: {diff:.2e}")

if __name__ == "__main__":
    # Test Baseline & Previous Attention Tasks
    test_causality("standard")
    test_causality("mqa")
    test_causality("linear")
    test_causality("standard", use_conv=True)
    
    # Test AFT Bonus Variants
    print("\n[AFT BONUS TEST]")
    test_causality("aft", "simple")
    test_causality("aft", "full")
    test_causality("aft", "local")
    test_causality("aft", "conv")
