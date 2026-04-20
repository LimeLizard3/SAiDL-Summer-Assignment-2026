import torch
import numpy as np
import gymnasium as gym
from td3 import TD3
from model import Normalizer
from collections import deque
import matplotlib.pyplot as plt
import os
from delayed_reward_wrapper import DelayedRewardWrapper

def recover_normalizer(policy, env_name, iterations=2500):
    """
    Since training stats weren't saved, we run the agent for a few steps 
    to re-calibrate the Normalizer's mean and std.
    """
    print(f"Recovering Normalizer stats...")
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    normalizer = Normalizer(shape=(state_dim,))
    
    state, _ = env.reset()
    
    # Standard history for Transformers if needed
    seq_len = getattr(policy, 'seq_len', 1) #'seq_len' is what we're searching for from policy (model). If it can't find returns 1
    state_history = deque([np.zeros(state_dim) for _ in range(seq_len)], maxlen=seq_len)
    action_history = deque([np.zeros(env.action_space.shape[0]) for _ in range(seq_len)], maxlen=seq_len)

    for _ in range(iterations):
        state_norm = normalizer(state, update=True)
        
        if policy.use_transformer:
            state_history.append(state_norm)
            action = policy.select_action(state_norm, state_history, action_history)
            action_history.append(action)
        else:
            action = policy.select_action(state_norm)
            
        state, _, terminated, truncated, _ = env.step(action) #Ignoring reward and info directories
        if terminated or truncated:
            state, _ = env.reset()
            state_history = deque([np.zeros(state_dim) for _ in range(seq_len)], maxlen=seq_len)
            action_history = deque([np.zeros(env.action_space.shape[0]) for _ in range(seq_len)], maxlen=seq_len)
            
    return normalizer

def evaluate_robustness(policy, env_name, normalizer, episodes=10, mode="clean"):
    """
    Evaluates agent under different environmental stress levels.
    """
    env = gym.make(env_name)
    if mode == "delayed":
        env = DelayedRewardWrapper(env, k=10) # Testing with standard K=10 delay
        
    state_dim = env.observation_space.shape[0]
    seq_len = getattr(policy, 'seq_len', 1)
    
    total_reward = 0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        state_history = deque([np.zeros(state_dim) for _ in range(seq_len)], maxlen=seq_len)
        action_history = deque([np.zeros(env.action_space.shape[0]) for _ in range(seq_len)], maxlen=seq_len)
        
        while not done:
            # 1. Apply Stressor (Partial Observability or Noise)
            obs = state.copy() #Disconnected copy of state, if we messed with state directly we may mess up the env
            
            if mode == "noisy":
                # Add 5% Gaussian noise to all sensors
                obs += np.random.normal(0, 0.05, size=state_dim)
                #Mean = 0 ensures that it doesn't skew readings consistently up or down
                #This ensures that the errors "cancel out" and that it's telling the truth
                #If it wasn't 0, it'd be like using a broken instrument
            elif mode == "partial":
                # MASK VELOCITIES (Sensors 6 to 10)
                obs[6:] = 0 
                #What exactly we're masking depends on the env used, here it's just velocities
                
            # 2. Normalize and Predict
            state_norm = normalizer(obs, update=False)
            
            if policy.use_transformer:
                state_history.append(state_norm)
                action = policy.select_action(state_norm, state_history, action_history)
                action_history.append(action)
            else:
                action = policy.select_action(state_norm)
                
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
    return total_reward / episodes

def run_analysis():
    env_name = "Hopper-v5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Agents
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    configs = [
        {"name": "MLP (Baseline)", "model_path": "./models/TD3_Hopper-v5_0", "transformer": False, "seq_len": 1},
        {"name": "Transformer (L=16)", "model_path": "./models/TD3_Transformer_L16_S0", "transformer": True, "seq_len": 16},
        {"name": "Transformer (L=32)", "model_path": "./models/TD3_Transformer_L32_S0", "transformer": True, "seq_len": 32},
    ]
    
    results = {cfg["name"]: [] for cfg in configs}
    modes = ["clean", "noisy", "partial", "delayed"]
    
    for cfg in configs:
        print(f"\nAnalyzing: {cfg['name']}")
        
        # Load Model
        policy = TD3(state_dim, action_dim, max_action, device, 
                     use_transformer=cfg['transformer'], seq_len=cfg['seq_len'])
        policy.load(cfg['model_path'])
        
        # Load/Prepare Normalizer
        normalizer = Normalizer(shape=(state_dim,))
        norm_path = cfg['model_path'] + "_normalizer.npz"
        if os.path.exists(norm_path):
            print(f"  Loading Normalizer from {norm_path}")
            normalizer.load(cfg['model_path'])
        else:
            print("  No saved normalizer found. Using identity (Standard MLP behavior).")
            
        model_results = []
        for mode in modes:
            score = evaluate_robustness(policy, env_name, normalizer, episodes=10, mode=mode)
            print(f"  Mode: {mode.upper()} -> Score: {score:.2f}")
            results[cfg['name']].append(score)

    # 3. Plotting
    plt.figure(figsize=(12, 7))
    x = np.arange(len(modes))
    width = 0.25
    
    for i, (name, scores) in enumerate(results.items()):
        plt.bar(x + i*width, scores, width, label=name)
        
    plt.title("Robustness Analysis: Memory vs. Reflex (Hopper-v5)")
    plt.xticks(x + width, [m.upper() for m in modes])
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if not os.path.exists("./analysis"):
        os.makedirs("./analysis")
        
    plt.savefig("./analysis/robustness_results.png")
    print("\n[SUCCESS] Analysis complete. Results saved to ./analysis/robustness_results.png")
    # plt.show()

if __name__ == "__main__":
    run_analysis()
