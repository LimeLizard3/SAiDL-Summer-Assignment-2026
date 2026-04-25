import torch
import numpy as np
import gymnasium as gym
from td3 import TD3
from model import Normalizer
from collections import deque
import matplotlib.pyplot as plt
import os

def run_attribution_diagnostics(model_path, env_name="Hopper-v5", seq_len=32, episodes=3):
    print(f"Running Advanced Attention Attribution...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    policy = TD3(state_dim, action_dim, max_action, device, use_transformer=True, seq_len=seq_len)
    policy.load(model_path)
    
    normalizer = Normalizer(shape=(state_dim,))
    normalizer.load(model_path)
    
    mean_raw_attn = np.zeros(seq_len)
    mean_attribution = np.zeros(seq_len)
    total_steps = 0
    
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        state_history = deque([np.zeros(state_dim) for _ in range(seq_len)], maxlen=seq_len)
        action_history = deque([np.zeros(action_dim) for _ in range(seq_len)], maxlen=seq_len)
        
        step_count = 0
        while not done:
            step_count += 1
            # FORCE MEMORY USAGE: Mask velocities
            obs = state.copy()
            obs[6:] = 0
            
            state_norm = normalizer(obs, update=False)
            state_history.append(state_norm)
            
            # Record a snapshot when the robot is in mid-stride
            if ep == 0 and step_count == 25:
                print("Capturing Stride Attribution Snapshot (Step 25)...")
                _, attn = policy.actor.select_action(state_norm, state_history, action_history, return_attn=True)
                mean_raw_attn = np.mean(attn[0, :, -1, :], axis=0)
                mean_attribution = policy.actor.get_attribution(state_history, action_history)
                total_steps = 1
                break # We just want this one perfect moment
            
            action = policy.select_action(state_norm, state_history, action_history)
            state, reward, terminated, truncated, _ = env.step(action)
            action_history.append(action)
            done = terminated or truncated
            
    mean_raw_attn /= total_steps
    mean_attribution /= total_steps
    return mean_raw_attn, mean_attribution

def plot_comparison(raw_data, attr_data, seq_len):
    plt.figure(figsize=(10, 6))
    timesteps = np.arange(-(seq_len-1), 1)
    
    plt.plot(timesteps, raw_data, label='Raw Attention Weights (Task 3)', color='gray', alpha=0.5, linestyle='--')
    plt.plot(timesteps, attr_data, label='Chefer Attribution (Task 2e)', color='crimson', linewidth=2.5)
    
    # Highlight the 'Decoupling' of Raw vs Attribution
    plt.fill_between(timesteps, raw_data, attr_data, color='crimson', alpha=0.1, label='Attribution Filtering')
    
    plt.title("Attention Attribution (Chefer et al.) vs. Raw Attention", fontsize=14)
    plt.xlabel("Relative Timestep (History <--- 0 ---> Present)")
    plt.ylabel("Contribution Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if not os.path.exists("./analysis"):
        os.makedirs("./analysis")
    plt.savefig("./analysis/attribution_comparison_2e.png", dpi=300)
    print("[SUCCESS] Advanced attribution analysis saved to ./analysis/attribution_comparison_2e.png")

if __name__ == "__main__":
    # Switching to the 'Robust Specialist' to see memory in action
    model_path = "./models/TD3_Transformer_Robust_stable_best"
    if os.path.exists(model_path + "_actor"):
        raw, attr = run_attribution_diagnostics(model_path)
        plot_comparison(raw, attr, 32)
    else:
        print("Champion model not found.")
