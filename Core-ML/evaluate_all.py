import torch
import torch.nn as nn
import time
import math
import os
from config import TransformerConfig
from model import TransformerLM
from data import get_dataloaders

def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6 # MB
    return 0

def evaluate_variant(attn_type, pos_type, device, train_loader, val_loader, use_conv=False, conv_type="pre_attention"):
    """
    Tests one specific combination of attention and position.
    Returns: (PPL, Tokens/Sec, Peak_VRAM_MB)
    """
    config = TransformerConfig()
    config.attention_type = attn_type
    config.pos_type = pos_type
    config.use_conv = use_conv
    config.conv_type = conv_type
    
    # We use a small model and short sequence to keep the test fast
    config.d_model = 128
    config.n_heads = 4
    config.n_layers = 2
    config.max_seq_len = 512
    
    torch.cuda.reset_peak_memory_stats()
    model = TransformerLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # 1. Measurement: Training Speed (10 steps)
    model.train()
    start_time = time.time()
    total_tokens = 0
    
    # Only run first few steps to get a "warm" measurement
    for i, (x, y) in enumerate(train_loader):
        if i >= 10: break
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        
        total_tokens += x.numel()
        
    duration = time.time() - start_time
    tokens_per_sec = total_tokens / (duration + 1e-9)
    peak_vram = get_gpu_memory()

    # 2. Measurement: Perplexity (on Validation set)
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= 5: break # Fast evaluation
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            count += 1
            
    avg_loss = total_loss / count
    ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    
    return ppl, tokens_per_sec, peak_vram

def run_full_diagnostic():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- STARTING GLOBAL DIAGNOSTIC ON {str(device).upper()} ---")
    
    # Setup Data (WikiText-2)
    # Note: If tokenizing for the first time, this might take a second.
    try:
        train_loader, val_loader, _ = get_dataloaders(batch_size=4, seq_len=128)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    attn_variants = ["standard", "mqa", "sliding_window", "linear"]
    pos_variants = ["absolute", "rope", "alibi"]
    
    results = []
    
    for attn in attn_variants:
        for pos in pos_variants:
            # Linear Attention only supports RoPE in our architecture
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
                
                
    # [Task 4] Test the "Winning" Hybrid Combinations
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

    # FINAL SUMMARY TABLE
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
