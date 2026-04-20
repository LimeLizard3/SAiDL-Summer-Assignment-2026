import torch
import numpy as np
import gymnasium as gym
from td3 import TD3
from model import Normalizer
from collections import deque
import matplotlib.pyplot as plt
import os
#The whole point of this file is to see how Transformer survived robustness and why MLP didn't

def calculate_entropy(attn_weights):
    """Calculates Shannon entropy of attention weights."""
    # attn_weights shape: (Heads, Seq_Len)
    # Avoid log(0)
    epsilon = 1e-10
    entropy = -np.sum(attn_weights * np.log(attn_weights + epsilon), axis=-1) #Do this across the Seq_len (axis=-1)
    #If the AI looks at 1 frame with 100% attention, entropy = roughly 0
    #If the AI looks at all frames equally, entropy = roughly 1
    return np.mean(entropy) # Average entropy across heads

def run_diagnostics(model_path, env_name="Hopper-v5", seq_len=16, episodes=5, mode="clean"):
    print(f"Running Attention Diagnostics for {mode} mode...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    policy = TD3(state_dim, action_dim, max_action, device, use_transformer=True, seq_len=seq_len)
    policy.load(model_path)
    
    # Simple normalizer recovery
    normalizer = Normalizer(shape=(state_dim,))
    
    all_entropies = []
    mean_attn_weights = np.zeros(seq_len) #seq_len frames, seq_len attention weights
    total_steps = 0
    
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        state_history = deque([np.zeros(state_dim) for _ in range(seq_len)], maxlen=seq_len)
        action_history = deque([np.zeros(action_dim) for _ in range(seq_len)], maxlen=seq_len)
        
        while not done:
            obs = state.copy()
            if mode == "partial":
                obs[6:] = 0 # Hide velocities
            
            state_norm = normalizer(obs, update=True)
            state_history.append(state_norm)
            
            # 1. Get action and attention weights
            action, attn = policy.select_action(state_norm, state_history, action_history, return_attn=True)
            
            # attn shape: (1, Heads, L, L) that 1 is batch size (#No. of Robots being trained)
            # We care about the attention from the CURRENT token (last row) to the past
            # Shape: (Heads, L) This shows "Right now, in the current frame, how much attention is each had paying to the past frames?"
            current_attn = attn[0, :, -1, :]  #0 and -1 collapse their respective dimensions
            
            # 2. Track Entropy
            all_entropies.append(calculate_entropy(current_attn))
            
            # 3. Accumulated Mean Weights
            mean_attn_weights += np.mean(current_attn, axis=0)
            
            state, reward, terminated, truncated, _ = env.step(action)
            action_history.append(action)
            done = terminated or truncated
            total_steps += 1
            
    mean_attn_weights /= total_steps
    return mean_attn_weights, all_entropies

def plot_diagnostics(clean_data, partial_data, seq_len):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Mean Attention Weights
    timesteps = np.arange(-(seq_len-1), 1)
    ax1.plot(timesteps, clean_data[0], marker='o', label='Fully Observable')
    ax1.plot(timesteps, partial_data[0], marker='s', label='Hidden Velocity')
    ax1.set_title("Mean Attention Weights (Last Decision)")
    ax1.set_xlabel("Relative Timestep (0 = Current)")
    ax1.set_ylabel("Attention Weight")
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Entropy Distribution
    ax2.hist(clean_data[1], bins=30, alpha=0.5, label='Fully Observable')
    ax2.hist(partial_data[1], bins=30, alpha=0.5, label='Hidden Velocity')
    ax2.set_title("Attention Entropy Distribution")
    ax2.set_xlabel("Entropy H(At)")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    
    plt.tight_layout()
    if not os.path.exists("./analysis"):
        os.makedirs("./analysis")
    plt.savefig("./analysis/attention_analysis.png")
    print("[SUCCESS] Attention analysis saved to ./analysis/attention_analysis.png")
    plt.show()

if __name__ == "__main__":
    # Use our best performing model (L=16)
    model_path = "./models/TD3_Transformer_L16_S0"
    if os.path.exists(model_path + "_actor"):
        clean = run_diagnostics(model_path, mode="clean", seq_len=16)
        partial = run_diagnostics(model_path, mode="partial", seq_len=16)
        plot_diagnostics(clean, partial, 16)
    else:
        print(f"Model path {model_path} not found. Please train L=16 first.")
