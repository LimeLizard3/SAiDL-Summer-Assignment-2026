import gymnasium as gym
import torch
import numpy as np
import os
from td3 import TD3
from model import Normalizer
from replay_buffer import ReplayBuffer
import gymnasium as gym

# THE ROBUSTNESS SPRINT
# This script trains a Transformer on 'Hidden Velocities' 
# to force it to use history to calculate speed.

def train_robust(seed=0, steps=20000):
    env_name = "Hopper-v5"
    env = gym.make(env_name)
    
    # Set seeds
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We use L=32 for maximum memory power
    seq_len = 32
    policy = TD3(state_dim, action_dim, max_action, device, use_transformer=True, seq_len=seq_len)
    
    # LOAD CHAMPION (L=32 S0)
    model_path = "./models/TD3_Transformer_L32_S0"
    policy.actor.load_state_dict(torch.load(model_path + "_actor", map_location=device))
    print(f"Loaded Champion: {model_path}")
    
    # LOAD EXPERT NORMALIZER (VITAL)
    normalizer = Normalizer(shape=(state_dim,))
    normalizer.load(model_path)
    
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    
    state, _ = env.reset(seed=seed)
    state_history = []
    action_history = []
    
    # Mask Velocities (Indices 6 onwards)
    def mask_velocity(s):
        s_masked = s.copy()
        s_masked[6:] = 0
        return s_masked

    # Initialize histories with masked zeros
    for _ in range(seq_len):
        state_history.append(mask_velocity(np.zeros(state_dim)))
        action_history.append(np.zeros(action_dim))

    print(f"Starting Robustness Sprint ({steps} steps)...")
    
    for t in range(steps):
        # 1. Mask and Normalize observation
        obs_masked = mask_velocity(state)
        state_norm = normalizer(obs_masked, update=False)
        
        state_history.append(state_norm)
        if len(state_history) > seq_len:
            state_history.pop(0)
            
        # 2. Select Action
        action = policy.select_action(state_norm, state_history, action_history)
        
        # 3. Step Environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done_bool = float(terminated or truncated)
        
        # 4. Mask and Store in buffer
        next_obs_masked = mask_velocity(next_state)
        next_state_norm = normalizer(next_obs_masked, update=False)
        
        replay_buffer.add(state_norm, action, next_state_norm, reward, done_bool)
        
        state = next_state
        action_history.append(action)
        if len(action_history) > seq_len:
            action_history.pop(0)

        # 5. Train
        if t > 2000:
            policy.train(replay_buffer, batch_size=64)

        if terminated or truncated:
            state, _ = env.reset()
            state_history = []
            action_history = []
            for _ in range(seq_len):
                state_history.append(mask_velocity(np.zeros(state_dim)))
                action_history.append(np.zeros(action_dim))

        if (t + 1) % 5000 == 0:
            print(f"Step {t+1}/{steps} complete.")

    # SAVE ROBUST MODEL
    save_path = "./models/TD3_Transformer_Robust"
    if not os.path.exists("./models"):
        os.makedirs("./models")
    torch.save(policy.actor.state_dict(), save_path + "_actor")
    normalizer.save(save_path)
    print(f"[SUCCESS] Robust Model saved to {save_path}")

if __name__ == "__main__":
    train_robust()
