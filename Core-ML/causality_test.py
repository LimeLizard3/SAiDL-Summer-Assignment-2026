import torch
import torch.nn as nn
from config import TransformerConfig
from model import TransformerLM

def test_causality(attention_type="standard", aft_mode="full", use_conv=False):
    desc = f"{attention_type.upper()}"
    if attention_type == "aft":
        desc += f" ({aft_mode})"
    if use_conv:
        desc += " + CONV"
        
    print(f"\n--- Testing Causality for: {desc} ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = TransformerConfig()
    config.attention_type = attention_type
    config.aft_mode = aft_mode
    config.use_conv = use_conv
    config.n_layers = 1 # Single layer is enough for flow check
    
    model = TransformerLM(config).to(device)
    model.eval()
    
    # 1. Create a base sequence X
    seq_len = 20
    x1 = torch.randint(0, config.vocab_size, (1, seq_len)).to(device)
    #0 to vocab_size is the range. (1,seq_len) represents 1 row seq_len columns
    
    # 2. Create identical sequence until split_idx
    split_idx = 10
    x2 = x1.clone()
    # Change ONLY the token at split_idx
    x2[0, split_idx] = (x1[0, split_idx] + 7) % config.vocab_size
    
    with torch.no_grad():
        output1 = model(x1)
        output2 = model(x2)
        
    # 3. Check if outputs BEFORE split_idx are identical
    # If word 11 changing affects word 10, causality is broken.
    diff = torch.abs(output1[0, :split_idx] - output2[0, :split_idx]).max().item()
    
    if diff < 1e-4:
        print(f"  PASS: No leakage. Max diff: {diff:.2e}")
    else:
        print(f"  FAILED: Leakage detected. Max diff: {diff:.2e}")

    #This whole thing is to ensure that the model doesn't look at the future

if __name__ == "__main__":
    # Test Baseline & Previous Tasks
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
