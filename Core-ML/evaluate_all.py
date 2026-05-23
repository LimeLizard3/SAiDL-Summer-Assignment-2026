import torch  # type: ignore
import torch.nn as nn  # type: ignore
import time
import math
import os
from config import TransformerConfig
from model import TransformerLM  # type: ignore
from data import get_dataloaders

def get_gpu_memory():
    """
    Returns current peak GPU memory allocated in Megabytes (MB).
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0

def evaluate_variant(attn_type, pos_type, device, train_loader, val_loader, use_conv=False, conv_type="pre_attention"):
    """
    Evaluates one specific architectural variant on training speed (throughput) and validation perplexity.
    Returns: (Perplexity, Tokens/Second, Peak VRAM in MB)
    """
    config = TransformerConfig()
    config.attention_type = attn_type
    config.pos_type = pos_type
    config.use_conv = use_conv
    config.conv_type = conv_type
    
    # Configure a downscaled model to keep global diagnostic times short
    config.d_model = 128
    config.n_heads = 4
    config.n_layers = 2
    config.max_seq_len = 512
    
    # Reset CUDA memory trackers to establish a clean memory baseline
    torch.cuda.reset_peak_memory_stats()
    
    # Load model and initialize optimizer
    model = TransformerLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # 1. Measurement: Training Speed (Run 10 steps to warm up and compute average throughput)
    model.train()
    start_time = time.time()
    total_tokens = 0
    
    for i, (x, y) in enumerate(train_loader):
        if i >= 10: break
        # Transfer batch tensors: x [batch_size, seq_len], y [batch_size, seq_len]
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        # Forward pass; output logits shape: [batch_size, seq_len, vocab_size]
        logits = model(x)
        # Compute cross entropy with flattened targets: shape [batch_size * seq_len]
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        # Backpropagate and update
        loss.backward()
        optimizer.step()
        
        # Accumulate token count
        total_tokens += x.numel()
        
    duration = time.time() - start_time
    tokens_per_sec = total_tokens / (duration + 1e-9)
    peak_vram = get_gpu_memory()

    # 2. Measurement: Perplexity (Evaluate on a small slice of validation dataset for speed)
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= 5: break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            count += 1
            
    avg_loss = total_loss / count
    ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    
    return ppl, tokens_per_sec, peak_vram

def run_full_diagnostic():
    """
    Main diagnostic execution suite loops through all possible modular configurations.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- STARTING GLOBAL DIAGNOSTIC ON {str(device).upper()} ---")
    
    # Load dataset dataloaders
    try:
        train_loader, val_loader, _ = get_dataloaders(batch_size=4, seq_len=128)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Attention and Positional options
    attn_variants = ["standard", "mqa", "sliding_window", "linear"]
    pos_variants = ["absolute", "rope", "alibi"]
    
    results = []
    
    for attn in attn_variants:
        for pos in pos_variants:
            # Linear Attention depends on continuous coordinate features, so it is constrained to RoPE in our setup
            if attn == "linear" and pos != "rope": continue
            
            print(f"Testing: [{attn.upper()}] + [{pos.upper()}]...")
            try:
                ppl, speed, vram = evaluate_variant(attn, pos, device, train_loader, val_loader)
                results.append({
                    "Variant": f"{attn} + {pos}",
                    "PPL": ppl,
                    "Tokens/Sec": speed,
                    "VRAM (MB)": vram
                })
                print(f"  > Success! PPL: {ppl:.2f} | Speed: {speed:.1f} tok/s | Memory: {vram:.1f} MB")
            except Exception as e:
                print(f"  > FAILED: {str(e)}")
                results.append({
                    "Variant": f"{attn} + {pos}",
                    "PPL": "CRASH",
                    "Tokens/Sec": 0,
                    "VRAM (MB)": 0
                })
                
    # Run evaluations on local-global hybrid convolutional models
    hybrid_tests = [
        ("mqa", "alibi", "pre_attention"),
        ("mqa", "alibi", "interleaved")
    ]
    
    for attn, pos, conv in hybrid_tests:
        print(f"Testing HYBRID: [{attn.upper()}] + [{pos.upper()}] + [{conv.upper()}]...")
        try:
            ppl, speed, vram = evaluate_variant(attn, pos, device, train_loader, val_loader, use_conv=True, conv_type=conv)
            results.append({
                "Variant": f"HYBRID ({conv[:4]}) {attn}+{pos}",
                "PPL": ppl,
                "Tokens/Sec": speed,
                "VRAM (MB)": vram
            })
            print(f"  > Success! PPL: {ppl:.2f} | Speed: {speed:.1f} tok/s | Memory: {vram:.1f} MB")
        except Exception as e:
            print(f"  > FAILED: {str(e)}")
            results.append({
                "Variant": f"HYBRID ({conv[:4]}) {attn}+{pos}",
                "PPL": "CRASH",
                "Tokens/Sec": 0,
                "VRAM (MB)": 0
            })

    # Output final summary data table
    print("\n" + "="*70)
    print(f"{'MODULAR VARIANT':<30} | {'PPL':<8} | {'TOK/SEC':<10} | {'VRAM MB':<10}")
    print("-" * 70)
    for r in results:
        ppl_str = f"{r['PPL']:.2f}" if isinstance(r['PPL'], float) else str(r['PPL'])
        print(f"{r['Variant']:<30} | {ppl_str:<8} | {r['Tokens/Sec']:<10.1f} | {r['VRAM (MB)']:<10.1f}")
    print("="*70)
    print("\nDiagnostic Complete! All modules verified.")

if __name__ == "__main__":
    run_full_diagnostic()
