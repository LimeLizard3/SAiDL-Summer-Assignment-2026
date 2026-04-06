import torch
import torch.nn as nn
from config import TransformerConfig
from model import TransformerLM

def test_causality(conv_type="pre_attention"):
    print(f"\n--- Testing Causality for: {conv_type.upper()} ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = TransformerConfig()
    config.use_conv = True
    config.conv_type = conv_type
    config.n_layers = 2
    
    model = TransformerLM(config).to(device)
    model.eval()
    
    # 1. Create a base sequence X
    seq_len = 10
    x1 = torch.randint(0, config.vocab_size, (1, seq_len)).to(device)
    
    # 2. Create a modified sequence X2
    # It is identical to X1 for the first 5 words, but the 6th word is different
    split_idx = 5
    x2 = x1.clone()
    x2[0, split_idx] = (x2[0, split_idx] + 1) % config.vocab_size # % is there to safeguard the math
    
    with torch.no_grad():
        output1 = model(x1)
        output2 = model(x2)
        
    # 3. Check if the outputs for the first 5 words are identical
    # If they are different, it means word 6 influenced word 5 (FUTURE LEAKAGE!)
    diff = torch.abs(output1[0, :split_idx] - output2[0, :split_idx]).max().item()
    
    if diff < 1e-5:
        print(f"  PASS: No future leakage detected. Max difference: {diff:.2e}")
    else:
        print(f"  FAILED! Future leakage detected! Max difference: {diff:.2e}")
        print(f"  This means the convolution is looking into the future.")

if __name__ == "__main__":
    test_causality("pre_attention")
    test_causality("interleaved")
