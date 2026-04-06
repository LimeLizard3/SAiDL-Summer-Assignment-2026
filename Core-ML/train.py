import sys
import subprocess

def auto_install():
    packages = ["torch", "datasets", "tiktoken"]
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing missing package: {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

auto_install()

import torch
import torch.nn.functional as F
import time
import math
from data import get_dataloaders
from model import TransformerLM
from config import TransformerConfig

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += y.numel()
    
    end_time = time.time()
    duration = end_time - start_time
    inference_throughput = total_tokens / duration if duration > 0 else 0
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    return avg_loss, perplexity, inference_throughput

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = TransformerConfig()
    print(f"Running Experiment: Attention={config.attention_type} | SeqLen={config.max_seq_len}")
    
    train_loader, val_loader, vocab_size = get_dataloaders(
        seq_len=config.max_seq_len, 
        batch_size=config.batch_size
    )
    config.vocab_size = vocab_size
    
    model = TransformerLM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    epochs = config.epochs
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        tokens_processed = 0
        
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            tokens_processed += y.numel()
            
            if i > 0 and i % 50 == 0:
                elapsed = time.time() - start_time
                throughput = tokens_processed / elapsed
                avg_train_loss = total_loss / 50
                train_ppl = math.exp(avg_train_loss) if avg_train_loss < 20 else float('inf')
                
                mem_reserved = 0
                if torch.cuda.is_available():
                    mem_reserved = torch.cuda.max_memory_reserved() / (1024**2)
                
                print(f"Epoch {epoch+1} | Step {i} | Loss: {avg_train_loss:.4f} | PPL: {train_ppl:.2f} | Train Speed: {throughput:.2f} tok/s | Mem: {mem_reserved:.1f} MB")
                
                total_loss = 0.0
                start_time = time.time()
                tokens_processed = 0
                
        # End of epoch validation
        val_loss, val_ppl, val_speed = evaluate(model, val_loader, device)
        
        peak_mem = 0
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_reserved() / (1024**2)
            
        print(f"\n--- Experiment Results: {config.attention_type} ---")
        print(f"Validation PPL: {val_ppl:.2f}")
        print(f"Inference Speed: {val_speed:.2f} tokens/sec")
        print(f"Peak GPU Memory: {peak_mem:.1f} MB")
        print(f"-------------------------------------------\n")

if __name__ == '__main__':
    main()
