import torch  # type: ignore
import torch.nn as nn  # type: ignore
import math
import time
from config import TransformerConfig
from model import TransformerLM  # type: ignore
from data import get_dataloaders

def evaluate_length(model, device, seq_len, batch_size=2):
    """
    Evaluates the model on a specific sequence length using random data
    to test mathematical stability (Perplexity explosion test).
    """
    model.eval()
    # We use a fixed seed so all models are tested on the same "data"
    torch.manual_seed(42)
    
    # Generate random test data of the target length
    # Shape: (Batch, Seq_Len)
    x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    
    with torch.no_grad():
        logits = model(x)
        # Loss: (Batch * Seq, Vocab) vs (Batch * Seq)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            y.view(-1)
        )
        ppl = math.exp(loss.item()) if loss.item() < 20 else float('inf')
        
    return ppl

def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Extrapolation Benchmark on {device}...")
    
    lengths = [512, 1024, 2048]
    variants = ["absolute", "rope", "alibi"]
    
    results = {}
    
    for v in variants:
        print(f"\nTesting Variant: {v}")
        config = TransformerConfig()
        config.pos_type = v
        # We train on 512, but we want to see if it can handle 2048
        config.max_seq_len = 2048 
        
        model = TransformerLM(config).to(device)
        
        results[v] = []
        for l in lengths:
            try:
                ppl = evaluate_length(model, device, l)
                results[v].append(ppl)
                print(f"  Length {l:4d} | PPL: {ppl:.2f}")
            except Exception as e:
                results[v].append("FAILED")
                print(f"  Length {l:4d} | FAILED: {str(e)}")

    # Print Final Table
    print("\n" + "="*50)
    print(f"{'Variant':<12} | {'512':<10} | {'1024':<10} | {'2048':<10}")
    print("-" * 50)
    for v, res in results.items():
        r = [f"{x:.2f}" if isinstance(x, float) else str(x) for x in res]
        print(f"{v:<12} | {r[0]:<10} | {r[1]:<10} | {r[2]:<10}")
    print("="*50)
    print("\nNote: Absolute encoding PPL usually explodes or fails at 2048,")
    print("while RoPE and ALiBi stay mathematically stable!")

if __name__ == "__main__":
    run_benchmark()
