import torch
import numpy as np
import gymnasium as gym
from td3 import TD3
from model import Normalizer
from collections import deque
import matplotlib.pyplot as plt
import os
#The whole point of this file is to see how Transformer survived robustness and why MLP didn't

def calculate_entropy_per_step(attn_weights): #Calculates Shannon entropy for a single sample
    """Calculates Shannon entropy of attention weights for the last query."""
    # attn_weights shape: (Heads, Seq_Len)
    epsilon = 1e-10
    entropy = -np.sum(attn_weights * np.log(attn_weights + epsilon), axis=-1)
    return np.mean(entropy)

def run_diagnostics(model_path, env_name="Hopper-v5", seq_len=32, episodes=3, mode="clean"):
    print(f"Running Attention Diagnostics for {mode} mode...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    policy = TD3(state_dim, action_dim, max_action, device, use_transformer=True, seq_len=seq_len)
    # Inference-only load (we only need the actor for diagnostics)
    policy.actor.load_state_dict(torch.load(model_path + "_actor", map_location=device))
    
    normalizer = Normalizer(shape=(state_dim,))
    normalizer.load(model_path) # Load the actual expert normalizer
    
    mean_attn_weights = np.zeros(seq_len)
    episode_entropies = [] # List of lists (one per episode)
    total_steps = 0
    
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        state_history = deque([np.zeros(state_dim) for _ in range(seq_len)], maxlen=seq_len)
        action_history = deque([np.zeros(action_dim) for _ in range(seq_len)], maxlen=seq_len)
        curr_ep_entropies = []
        
        while not done:
            obs = state.copy()
            if mode == "partial":
                obs[6:] = 0 # Hide velocities (Partial Observability)
            
            state_norm = normalizer(obs, update=False) # Don't update during eval
            state_history.append(state_norm)
            
            action, attn = policy.select_action(state_norm, state_history, action_history, return_attn=True)
            
            # attn shape: (1, Heads, L, L)
            current_attn = attn[0, :, -1, :] # (Heads, L)
            
            curr_ep_entropies.append(calculate_entropy_per_step(current_attn))
            mean_attn_weights += np.mean(current_attn, axis=0)
            
            state, reward, terminated, truncated, _ = env.step(action)
            action_history.append(action)
            done = terminated or truncated
            total_steps += 1
            
        episode_entropies.append(curr_ep_entropies)
            
    mean_attn_weights /= total_steps
    return mean_attn_weights, episode_entropies

def plot_diagnostics(clean_data, partial_data, seq_len):
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Use standard contrasting colors (Blue for Clean, Orange for Robust)
    clean_color = '#1f77b4' 
    robust_color = '#ff7f0e'
    
    # 1. Mean Attention Weights Comparison
    timesteps = np.arange(-(seq_len-1), 1)
    ax1.plot(timesteps, clean_data[0], color=clean_color, linewidth=2, label='Fully Observable (Broad Focus)')
    ax1.plot(timesteps, partial_data[0], color=robust_color, linewidth=3, label='Robust Survivor (Surgical Focus)')
    
    # Annotate the spikes
    ax1.annotate('Deriving Velocity', xy=(0, partial_data[0][-1]), xytext=(-12, 0.08),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1))
    ax1.annotate('Deep Memory Anchor', xy=(-31, partial_data[0][0]), xytext=(-31, 0.04),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    ax1.set_title("The Transformer's Mind: Attention Strategy Comparison", fontsize=14, pad=15)
    ax1.set_xlabel("Relative Timestep (History <--- 0 ---> Present)", fontsize=11)
    ax1.set_ylabel("Attention Contribution", fontsize=11)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 2. Entropy Analysis
    clean_ep = clean_data[1][0]
    partial_ep = partial_data[1][0]
    
    ax2.plot(clean_ep[:250], color=clean_color, alpha=0.5, label='Clean Entropy')
    ax2.plot(partial_ep[:250], color=robust_color, alpha=0.8, label='Robust Entropy (Precision Focus)')
    
    ax2.set_title("Decision Sharpness: Attention Entropy Over Time", fontsize=14, pad=15)
    ax2.set_xlabel("Steps in Episode", fontsize=11)
    ax2.set_ylabel("Entropy H(At)", fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    if not os.path.exists("./analysis"):
        os.makedirs("./analysis")
    plt.savefig("./analysis/attention_analysis.png", dpi=300)
    print("[SUCCESS] Scientific attention analysis saved to ./analysis/attention_analysis.png")

if __name__ == "__main__":
    # Use L=32 Champion for Clean
    model_path = "./models/TD3_Transformer_L32_S0"
    # Use L=32 Robust Specialist for Partial
    robust_path = "./models/TD3_Transformer_Robust"
    
    seq_len = 32
    if os.path.exists(model_path + "_actor") and os.path.exists(robust_path + "_actor"):
        clean = run_diagnostics(model_path, mode="clean", seq_len=seq_len)
        partial = run_diagnostics(robust_path, mode="partial", seq_len=seq_len)
        plot_diagnostics(clean, partial, seq_len)
    else:
        print(f"Required models not found.")
