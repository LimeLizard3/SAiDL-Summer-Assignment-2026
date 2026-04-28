import os
import torch
import numpy as np
import gymnasium as gym
from collections import deque
from model import Normalizer
from td3 import TD3
import matplotlib.pyplot as plt

def evaluate_combined(policy, env_name, normalizer, episodes=20, delay_K=10):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    seq_len = policy.seq_len
    
    total_reward = 0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        
        # Transformer Memory
        state_history = deque([np.zeros(state_dim) for _ in range(seq_len)], maxlen=seq_len)
        action_history = deque([np.zeros(action_dim) for _ in range(seq_len)], maxlen=seq_len)
        
        # 1. Delay Buffer: The robot sees the world as it was K steps ago
        # We initialize with the starting state to prevent a 'Black Hole' at step 0
        obs_buffer = deque([state.copy() for _ in range(delay_K)], maxlen=delay_K)
        
        while not done:
            # Add current state to buffer and get the 'Old' observation from K steps ago
            obs_buffer.append(state.copy())
            obs = obs_buffer[0] # This is the state from K steps ago (Observation Delay)
            
            # 2. Stressors: Mask Velocity + Noise (Relative to Sensor Range)
            obs[6:] = 0 # Blindfold (Partial Observability)
            
            # Standardized Noise Scaler: Proportional to each sensor's natural variance
            noise_std = np.sqrt(normalizer.rms.var + 1e-8)
            obs += np.random.normal(0, 0.1, size=obs.shape) * noise_std 
            
            state_norm = normalizer(obs, update=False)
            state_history.append(state_norm)
            
            action = policy.select_action(state_norm, state_history, action_history)
            action_history.append(action)
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
    return total_reward / episodes

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./models/TD3_Transformer_L32_S0_stable_best"
    
    # Load Champion
    policy = TD3(11, 3, 1.0, device, use_transformer=True, seq_len=32, pos_encoding_type='learned')
    policy.load(model_path)
    
    normalizer = Normalizer(shape=(11,))
    normalizer.load(model_path)
    
    print(f"Running TRUE Triple-Threat Benchmark: Standardized Noise=0.1, Delay_K=10, Blindfold=True")
    score = evaluate_combined(policy, "Hopper-v5", normalizer, episodes=20, delay_K=10)
    print(f"Final Combined Challenge Score: {score:.2f}")
    
    # Save a simple report graph
    plt.figure(figsize=(8, 6))
    plt.bar(["Combined POMDP Challenge"], [score], color='red')
    plt.axhline(y=118, color='black', linestyle='--', label="Baseline Limit")
    plt.title("The Triple Threat: Final Performance (With True Lag)")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("Graphs/Combined_POMDP_Final.png")
    print("Graph saved to Graphs/Combined_POMDP_Final.png")

if __name__ == "__main__":
    run_benchmark()
