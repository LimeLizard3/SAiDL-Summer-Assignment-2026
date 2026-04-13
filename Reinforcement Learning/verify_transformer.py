import torch
import numpy as np
import gymnasium as gym
from td3 import TD3
from replay_buffer import ReplayBuffer
from model import Normalizer
from collections import deque

def test_transformer_short():
    env_name = "Hopper-v5"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] #shape[0] usually refers to length
    max_action = float(env.action_space.high[0]) #high[0] also refers to physical length/boundary
    device = torch.device("cpu") # Use CPU for quick test
    
    seq_len = 8
    print(f"Testing Transformer (L={seq_len}) initialization...")
    
    policy = TD3(state_dim, action_dim, max_action, device, use_transformer=True, seq_len=seq_len)
    buffer = ReplayBuffer(state_dim, action_dim, device=device)
    normalizer = Normalizer(shape=(state_dim,)) #The comma makes it a tuple, without it you'd get an error stating "int object isn't iterable"
    
    state, _ = env.reset()
    state_history = deque([np.zeros(state_dim) for _ in range(seq_len)], maxlen=seq_len)
    action_history = deque([np.zeros(action_dim) for _ in range(seq_len)], maxlen=seq_len)
    
    # Run for 20 steps to check shapes and logic
    for t in range(20):
        state_norm = normalizer(state)
        state_history.append(state_norm)
        
        action = policy.select_action(state_norm, list(state_history), list(action_history))
        next_state, reward, terminated, truncated, _ = env.step(action)
        action_history.append(action)
        
        buffer.add(state, action, next_state, reward, float(terminated))
        state = next_state
        
        if t >= seq_len:
            # Test training logic only after we have enough data for a sequence
            policy.train(buffer, batch_size=4)
            
    print("Transformer Integration Verified!")

if __name__ == "__main__":
    test_transformer_short()
