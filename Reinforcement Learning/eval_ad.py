import os
# pyrefly: ignore [missing-import]
import gymnasium as gym
# pyrefly: ignore [missing-import]
from gymnasium import spaces
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import numpy as np
from ad_model import ADTransformer
from td3 import TD3 # for comparison if needed

def evaluate_ad(model=None, model_path="ad_transformer.pth", env_name="Hopper-v4", num_episodes=10, seq_len=100):
    env = gym.make(env_name)
    assert isinstance(env.observation_space, spaces.Box)
    assert isinstance(env.action_space, spaces.Box)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model is None:
        model = ADTransformer(state_dim, action_dim)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
    
    model.eval() #Turns off dropout, etc. as we are now testing,

    total_rewards = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        
        # Buffers for context
        states_buffer = torch.zeros((1, seq_len, state_dim), device=device)
        actions_buffer = torch.zeros((1, seq_len, action_dim), device=device)
        rewards_buffer = torch.zeros((1, seq_len, 1), device=device)
        timesteps_buffer = torch.arange(seq_len, device=device).unsqueeze(0)
        
        step_idx = 0
        while not done:
            # Update buffers with current state
            # Shift left and add new state
            if step_idx < seq_len:
                states_buffer[0, step_idx] = torch.from_numpy(obs).float().to(device)
                curr_idx = step_idx
            else:
                states_buffer = torch.cat([states_buffer[:, 1:, :], torch.from_numpy(obs).float().to(device).view(1, 1, -1)], dim=1) #Our sliding window: Drops the oldest frame and glues the newest frame to the tail
                actions_buffer = torch.roll(actions_buffer, -1, dims=1) #Rolls all the data by 1 step, and takes the oldest one and puts it at the tail
                rewards_buffer = torch.roll(rewards_buffer, -1, dims=1)
                curr_idx = seq_len - 1 #Ignore the last one as that's the oldest one
            
            # Get action from model
            with torch.no_grad():
                # We need to pass the full buffers. 
                # The model's forward handles the whole sequence but we only care about the last prediction.
                # In AD, we predict a_t from s_0, a_0, r_0 ... s_t
                # Our model's forward does: x = stack(s, a, r) -> transformer -> predict_action(x_s)
                # So it predicts a_i from s_i (and previous context)
                
                # Note: For the very first step, actions and rewards are 0.
                pred_actions = model(states_buffer, actions_buffer, rewards_buffer, timesteps_buffer)
                action = pred_actions[0, curr_idx].cpu().numpy()
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update buffers with action and reward
            if step_idx < seq_len:
                actions_buffer[0, step_idx] = torch.from_numpy(action).to(device)
                rewards_buffer[0, step_idx] = torch.tensor([reward], device=device)
            else:
                actions_buffer[0, -1] = torch.from_numpy(action).to(device) #Overwriting the old data at the tail, and replacing it with the most recent data
                rewards_buffer[0, -1] = torch.tensor([reward], device=device)

            obs = next_obs
            episode_reward += float(reward)
            step_idx += 1
            
        total_rewards.append(episode_reward)
        print(f"Episode {ep+1}: Reward = {episode_reward:.2f}")

    print(f"\nAverage Reward over {num_episodes} episodes: {np.mean(total_rewards):.2f}")
    return total_rewards

if __name__ == "__main__":
    if os.path.exists("ad_transformer.pth"):
        evaluate_ad()
    else:
        print("Model file not found. Please train the model first.")
