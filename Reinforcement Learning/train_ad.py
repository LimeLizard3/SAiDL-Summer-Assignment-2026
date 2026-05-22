# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.optim as optim
# pyrefly: ignore [missing-import]
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader, random_split
from ad_dataset import ADDataset
from ad_model import ADTransformer
import os
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
from eval_ad import evaluate_ad
from torch.cuda.amp import GradScaler, autocast

def train_ad():
    # Hyperparameters
    batch_size = 128
    accumulation_steps = 4
    lr = 1e-4
    epochs = 9
    seq_len = 100
    n_embd = 128
    n_layer = 4
    n_head = 4
    # Locate directories relative to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "ad_dataset")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(">>> Starting dataset loading …")
    # Load dataset
    dataset = ADDataset(data_dir, seq_len=seq_len)
    print(f">>> Dataset loaded: {len(dataset)} sequences")
    
    # Split into 90% training and 10% validation
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Initialize model
    state_dim = 11 # Hopper-v4 state dim
    action_dim = 3 # Hopper-v4 action dim
    model = ADTransformer(state_dim, action_dim, n_layer=n_layer, n_head=n_head, n_embd=n_embd).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print(f"Starting training on {device}...")
    train_losses = []
    val_losses = []
    eval_rewards = []

    scaler = GradScaler()
    for epoch in range(epochs):
        # 1. Training Phase
        model.train()
        epoch_train_loss = 0
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            rewards = batch['rewards'].to(device)
            
            # Create dummy timesteps
            timesteps = torch.arange(seq_len).repeat(states.shape[0], 1).to(device)
            
            # Apply Scheduled Action Masking to prevent Causal Confusion
            p_mask = min(0.2 + 0.1 * epoch, 0.8)
            mask = (torch.rand(actions.shape[:-1], device=device) > p_mask).unsqueeze(-1).float()
            masked_actions = actions * mask #Getting action_dim back by broadcasting
            
            # Apply History Jitter as data augmentation to prevent overfitting
            jittered_states = states + torch.randn_like(states) * 0.01
            # Jitter only unmasked actions (preserve zeroed-out mask elements)
            jittered_actions = masked_actions + (torch.randn_like(actions) * 0.005) * mask
            jittered_rewards = rewards + torch.randn_like(rewards) * 0.01
            
            # Predict actions (Forward Pass) with mixed precision autocast
            with autocast(): #AMP
                pred_actions = model(jittered_states, jittered_actions, jittered_rewards, timesteps)
                # Loss: MSE between predicted actions and true actions (scaled by accumulation_steps)
                loss = criterion(pred_actions, actions) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            # Step optimizer every accumulation_steps batches
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_train_loss += loss.item() * accumulation_steps
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {i}/{len(train_loader)} | Train Loss: {loss.item() * accumulation_steps:.6f}")
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 2. Validation Phase
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                states = batch['states'].to(device)
                actions = batch['actions'].to(device)
                rewards = batch['rewards'].to(device)
                timesteps = torch.arange(seq_len).repeat(states.shape[0], 1).to(device)
                
                pred_actions = model(states, actions, rewards, timesteps)
                loss = criterion(pred_actions, actions)
                
                epoch_val_loss += loss.item()
                
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 3. Online Evaluation Phase
        print(f"Running online evaluation for Epoch {epoch+1}...")
        eval_episode_rewards = evaluate_ad(model=model, env_name="Hopper-v4", num_episodes=5, seq_len=seq_len)
        avg_eval_reward = sum(eval_episode_rewards) / len(eval_episode_rewards)
        eval_rewards.append(avg_eval_reward)
        
        print(f"Epoch {epoch+1} finished | Avg Train Loss: {avg_train_loss:.6f} | Avg Val Loss: {avg_val_loss:.6f} | Avg Eval Reward: {avg_eval_reward:.2f}")

    # Save model
    model_save_path = os.path.join(script_dir, "models", "ad_transformer.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss", color="#1e88e5", linewidth=2)
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss", color="#fb8c00", linestyle="--", linewidth=2)
    plt.title("Algorithm Distillation (AD) Offline Training & Validation Loss", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss (Action Prediction)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    
    graphs_dir = os.path.join(script_dir, "Graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    loss_graph_path = os.path.join(graphs_dir, "ad_training_loss.png")
    plt.savefig(loss_graph_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Report-ready training loss graph saved to {loss_graph_path}")

    # Plot dual-axis benchmark
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Left Axis: MSE Loss
    color_loss = "#1e88e5"
    ax1.set_xlabel("Epoch", fontsize=12, labelpad=10)
    ax1.set_ylabel("Offline MSE Loss (Action Prediction)", color=color_loss, fontsize=12, fontweight='bold')
    line1 = ax1.plot(range(1, epochs + 1), train_losses, label="Training Loss (Left)", color=color_loss, linewidth=2)
    line2 = ax1.plot(range(1, epochs + 1), val_losses, label="Validation Loss (Left)", color="#fb8c00", linestyle="--", linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Right Axis: Evaluation Reward
    ax2 = ax1.twinx()
    color_reward = "#2e7d32"
    ax2.set_ylabel("Online In-Context Evaluation Reward", color=color_reward, fontsize=12, fontweight='bold')
    line3 = ax2.plot(range(1, epochs + 1), eval_rewards, label="Evaluation Reward (Right)", color=color_reward, linewidth=2.5, marker='o', markersize=6)
    ax2.tick_params(axis='y', labelcolor=color_reward)
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [str(l.get_label()) for l in lines]
    ax1.legend(handles=lines, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=11, frameon=True) # type: ignore
    
    plt.title("AD Benchmark: Offline Training Loss vs. Online In-Context Evaluation Reward", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    benchmark_graph_path = os.path.join(graphs_dir, "ad_benchmark_results.png")
    plt.savefig(benchmark_graph_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Dual-axis benchmark graph saved to {benchmark_graph_path}")

if __name__ == "__main__":
    train_ad()
