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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Mean Attention Weights Comparison
    timesteps = np.arange(-(seq_len-1), 1)
    ax1.plot(timesteps, clean_data[0], marker='.', alpha=0.7, label='Fully Observable')
    ax1.plot(timesteps, partial_data[0], marker='.', alpha=0.7, label='Hidden Velocity')
    ax1.set_title("Mean Attention Weights (The Memory Map)")
    ax1.set_xlabel("Relative Timestep (0 = Current)")
    ax1.set_ylabel("Attention Contribution")
    ax1.legend()
    ax1.grid(True, linestyle='--')
    
    # 2. Entropy Over Time (Time-Series)
    # Plot the first episode of each for clarity
    clean_ep = clean_data[1][0]
    partial_ep = partial_data[1][1] if len(partial_data[1]) > 1 else partial_data[1][0]
    
    ax2.plot(clean_ep[:200], label='Fully Observable (Ep 1)', alpha=0.8)
    ax2.plot(partial_ep[:200], label='Hidden Velocity (Ep 1)', alpha=0.8)
    ax2.set_title("Attention Entropy $H(A_t)$ over Episode Steps")
    ax2.set_xlabel("Steps in Episode")
    ax2.set_ylabel("Entropy (Higher = More Broad)")
    ax2.legend()
    ax2.grid(True, linestyle='--')
    
    plt.tight_layout()
    if not os.path.exists("./analysis"):
        os.makedirs("./analysis")
    plt.savefig("./analysis/attention_analysis.png")
    print("[SUCCESS] Attention analysis saved to ./analysis/attention_analysis.png")

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
