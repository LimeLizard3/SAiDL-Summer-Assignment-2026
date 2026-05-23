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

def evaluate_config(config, device, train_loader, val_loader):
    """
    Runs training steps to measure throughput/memory, followed by validation steps to measure perplexity.
    """
    # Reset peak memory counter for clean isolated profiling
    torch.cuda.reset_peak_memory_stats()
    # Instantiate modular transformer from configuration
    model = TransformerLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Warmup loop: compiles operations and populates CUDA cache prior to measurement
    model.train()
    for i, (x, y) in enumerate(train_loader):
        if i >= 5: break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        # Flatten logits to [batch_size * seq_len, vocab_size] and targets to [batch_size * seq_len] for cross-entropy
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 1. Measurement: Execution Speed & Peak GPU Memory
    model.train()
    start_time = time.time()
    total_tokens = 0
    for i, (x, y) in enumerate(train_loader):
        if i >= 20: break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        # Accumulate token count (batch_size * seq_len)
        total_tokens += x.numel()
    
    duration = time.time() - start_time
    tokens_per_sec = total_tokens / (duration + 1e-9)
    peak_vram = get_gpu_memory()

    # 2. Measurement: Perplexity (PPL) on validation data slice
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= 10: break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            count += 1
            
    avg_loss = total_loss / count
    ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    
    return ppl, tokens_per_sec, peak_vram

def run_aft_benchmarks():
    """
    Loops through various AFT and standard configurations, running evaluations and outputting results.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- STARTING AFT BONUS BENCHMARKS ON {str(device).upper()} ---")
    
    # Load dataset
    try:
        train_loader, val_loader, _ = get_dataloaders(batch_size=4, seq_len=128)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # List of configurations to profile
    tests = [
        {"name": "Standard (Baseline)", "type": "standard", "pos": "absolute"},
        {"name": "Hybrid (Previous Best)", "type": "mqa", "pos": "alibi", "conv": True},
        {"name": "AFT-Simple", "type": "aft", "mode": "simple"},
        {"name": "AFT-Full", "type": "aft", "mode": "full"},
        {"name": "AFT-Local", "type": "aft", "mode": "local"},
        {"name": "AFT-Conv", "type": "aft", "mode": "conv"},
    ]
    
    results = []
    
    for t in tests:
        print(f"Benchmarking {t['name']}...")
        config = TransformerConfig()
        config.attention_type = str(t['type'])
        config.pos_type = str(t.get('pos', 'absolute'))
        config.use_conv = bool(t.get('conv', False))
        config.aft_mode = str(t.get('mode', 'full'))
        
        # Consistent scale across configurations
        config.d_model = 128
        config.n_heads = 4
        config.n_layers = 2
        config.max_seq_len = 512
        
        try:
            ppl, speed, vram = evaluate_config(config, device, train_loader, val_loader)
            results.append({
                "Variant": t['name'],
                "PPL": ppl,
                "Speed": speed,
                "VRAM": vram
            })
            print(f"  > PPL: {ppl:.2f} | Speed: {speed:.1f} tok/s | VRAM: {vram:.1f} MB")
        except Exception as e:
            print(f"  > FAILED: {str(e)}")
            results.append({"Variant": t['name'], "PPL": "FAIL", "Speed": 0, "VRAM": 0})

    # Save results to a Markdown file
    with open("aft_results.md", "w") as f:
        f.write("# AFT Bonus Benchmarking Results\n\n")
        f.write("| Variant | PPL | Speed (tok/s) | VRAM (MB) |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for r in results:
            ppl_str = f"{r['PPL']:.2f}" if isinstance(r['PPL'], float) else r['PPL']
            f.write(f"| {r['Variant']} | {ppl_str} | {r['Speed']:.1f} | {r['VRAM']:.1f} |\n")
            
    print("\nBenchmarks complete! Results saved to aft_results.md")

if __name__ == "__main__":
    run_aft_benchmarks()
