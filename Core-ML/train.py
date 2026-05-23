import sys
import subprocess

def auto_install():
    """
    Checks for essential dependencies and installs them if missing.
    """
    packages = ["torch", "datasets", "tiktoken"]
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing missing package: {pkg}...")
            # Execute pip install programmatically
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# Run automatic package installation checks
auto_install()

import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
import time
import math
from data import get_dataloaders
from model import TransformerLM  # type: ignore
from config import TransformerConfig

def evaluate(model, val_loader, device):
    """
    Evaluates the model on validation data, returning average loss, perplexity, and throughput.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()
    
    with torch.no_grad():
        for x, y in val_loader:
            # Transfer tensors to target device; shapes: x [batch_size, seq_len], y [batch_size, seq_len]
            x, y = x.to(device), y.to(device)
            # Forward pass; logits shape: [batch_size, seq_len, vocab_size]
            logits = model(x)
            # Calculate sum of token cross entropies to normalize later
            # Flattened logits: [batch_size * seq_len, vocab_size], Flattened targets: [batch_size * seq_len]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += y.numel() # Number of target tokens in batch
    
    # Calculate performance duration
    duration = time.time() - start_time
    inference_throughput = total_tokens / duration if duration > 0 else 0
    
    # Compute normalized validation metrics
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    return avg_loss, perplexity, inference_throughput

def main():
    # Setup execution device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration hyperparameters
    config = TransformerConfig()
    print(f"Running Experiment: Attention={config.attention_type} | SeqLen={config.max_seq_len}")
    
    # Fetch training and validation PyTorch dataloaders
    train_loader, val_loader, vocab_size = get_dataloaders(
        seq_len=config.max_seq_len, 
        batch_size=config.batch_size
    )
    config.vocab_size = vocab_size
    
    # Initialize the modular language model
    model = TransformerLM(config).to(device)
    # Initialize AdamW optimizer with standard weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    epochs = config.epochs
    
    # Main training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        tokens_processed = 0
        
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # Reset parameter gradients
            optimizer.zero_grad()
            # Forward pass; logits shape: [batch_size, seq_len, vocab_size]
            logits = model(x)
            # Compute cross entropy loss; targets flattened to [batch_size * seq_len]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            # Backward pass to calculate gradients
            loss.backward()
            # Gradient clipping to prevent exploding gradients (max norm: 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters
            optimizer.step()
            
            total_loss += loss.item()
            tokens_processed += y.numel()
            
            # Print metrics periodically (every 50 steps)
            if i > 0 and i % 50 == 0:
                elapsed = time.time() - start_time
                throughput = tokens_processed / elapsed
                avg_train_loss = total_loss / 50
                train_ppl = math.exp(avg_train_loss) if avg_train_loss < 20 else float('inf')
                
                # Fetch GPU memory utilization statistics if available
                mem_reserved = 0
                if torch.cuda.is_available():
                    mem_reserved = torch.cuda.max_memory_reserved() / (1024**2)
                
                print(f"Epoch {epoch+1} | Step {i} | Loss: {avg_train_loss:.4f} | PPL: {train_ppl:.2f} | Train Speed: {throughput:.2f} tok/s | Mem: {mem_reserved:.1f} MB")
                
                # Reset accumulators for next tracking window
                total_loss = 0.0
                start_time = time.time()
                tokens_processed = 0
                
        # Run evaluation at end of epoch
        val_loss, val_ppl, val_speed = evaluate(model, val_loader, device)
        
        # Log peak memory usage
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
