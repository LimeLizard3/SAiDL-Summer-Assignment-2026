import os
import torch
import numpy as np
import gymnasium as gym
from collections import deque
from td3 import TD3
from model import Normalizer
from replay_buffer import ReplayBuffer

def generate_teacher_data(env_name="Hopper-v5", steps=50000, seq_len=32):
    print(f"--- Generating SYNCED Teacher Data using L=32 Transformer Actor ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(env_name)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # 1. Load the Best Model (L=32)
    policy = TD3(state_dim, action_dim, max_action, device, use_transformer=True, seq_len=seq_len)
    model_path = "./models/TD3_Transformer_L32_S0"
    policy.actor.load_state_dict(torch.load(model_path + "_actor", map_location=device))
    
    # 2. Load the corresponding Normalizer (CRITICAL)
    normalizer = Normalizer(shape=(state_dim,))
    if os.path.exists(model_path + "_normalizer.npz"):
        print("Loading L=32 Normalizer stats...")
        normalizer.load(model_path)
    
    # 3. Initialize Teacher Buffer
    buffer = ReplayBuffer(state_dim, action_dim, device=device)
    
    state, _ = env.reset(seed=0)
    state_history = deque([np.zeros(state_dim) for _ in range(seq_len)], maxlen=seq_len)
    action_history = deque([np.zeros(action_dim) for _ in range(seq_len)], maxlen=seq_len)
    
    total_reward = 0
    
    for t in range(steps):
        # Normalize (this is what we use in THE BUFFER now)
        state_norm = normalizer(state, update=False)
        state_history.append(state_norm)
        
        # Policy action 
        action = policy.select_action(state_norm, state_history, action_history)
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        action_history.append(action)
        
        # REPAIR LINE: We save 'state_norm' into the buffer.
        # This ensures the Judge learns from the SAME data scale that the student uses,
        # preventing 'Sensory Mismatch' before the experiment even begins.
        next_state_norm = normalizer(next_state, update=False)
        done_bool = float(terminated) if (t % 1000 != 0) else 0
        buffer.add(state_norm, action, next_state_norm, reward, done_bool)
        
        state = next_state
        total_reward += reward
        
        if terminated or truncated:
            state, _ = env.reset()
            state_history = deque([np.zeros(state_dim) for _ in range(seq_len)], maxlen=seq_len)
            action_history = deque([np.zeros(action_dim) for _ in range(seq_len)], maxlen=seq_len)
            
        if (t + 1) % 10000 == 0:
            print(f"Synced Collection: {t+1}/{steps} steps. Avg Reward: {total_reward/(t+1):.2f}")

    # 4. Save the Sync-Corrected Buffer
    buffer_save_path = "./models/teacher_buffer"
    buffer.save(buffer_save_path)
    print(f"\n[SUCCESS] SYNCED Teacher Data saved to {buffer_save_path}_buffer.npz")

if __name__ == "__main__":
    generate_teacher_data()
