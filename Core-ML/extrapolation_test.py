import torch  # type: ignore
import torch.nn as nn  # type: ignore
import math
import time
from config import TransformerConfig
from model import TransformerLM  # type: ignore
from data import get_dataloaders

def evaluate_length(model, device, seq_len, batch_size=2):
    """
    Evaluates the model's loss and perplexity on a specific sequence length.
    Tests mathematical stability at long context sizes to identify perplexity explosion.
    """
    model.eval()
    # Set seed to ensure identical target generation for fair comparison across variants
    torch.manual_seed(42)
    
    # Generate random test data: shape [batch_size, seq_len] containing random token IDs
    x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    # Generate target labels: shape [batch_size, seq_len]
    y = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    
    with torch.no_grad():
        # Forward pass; logits output shape: [batch_size, seq_len, vocab_size]
        logits = model(x)
        
        # Flatten logits to [batch_size * seq_len, vocab_size] and targets to [batch_size * seq_len]
        # to calculate standard cross entropy loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            y.view(-1)
        )
        
        # Compute perplexity: PPL = exp(loss). Cap at inf if loss value is extremely large.
        ppl = math.exp(loss.item()) if loss.item() < 20 else float('inf')
        
    return ppl

def run_benchmark():
    """
    Runs sequence length extrapolation benchmark comparing absolute, RoPE, and ALiBi positional strategies.
    """
    # Select hardware device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Extrapolation Benchmark on {device}...")
    
    # Test lengths and variants
    lengths = [512, 1024, 2048]
    variants = ["absolute", "rope", "alibi"]
    
    # Dict to aggregate results
    results = {}
    
    for v in variants:
        print(f"\nTesting Variant: {v}")
        # Initialize configuration with custom position embedding type
        config = TransformerConfig()
        config.pos_type = v
        # Allow sequence capacity up to maximum benchmark sequence size
        config.max_seq_len = 2048 
        
        # Instantiate model instance
        model = TransformerLM(config).to(device)
        
        results[v] = []
        for l in lengths:
            try:
                # Calculate perplexity for the given context size
                ppl = evaluate_length(model, device, l)
                results[v].append(ppl)
                print(f"  Length {l:4d} | PPL: {ppl:.2f}")
            except Exception as e:
                # Handle out-of-memory or computation failures
                results[v].append("FAILED")
                print(f"  Length {l:4d} | FAILED: {str(e)}")

    # Print summary performance metrics table
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
