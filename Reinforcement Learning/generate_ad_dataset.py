import os
# pyrefly: ignore [missing-import]
import gymnasium as gym
# pyrefly: ignore [missing-import]
from gymnasium import spaces
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import numpy as np
import pickle
from td3 import TD3
from replay_buffer import ReplayBuffer

def generate_history(env_name="Hopper-v4", total_timesteps=1000000, save_dir="ad_dataset"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    env = gym.make(env_name)
    assert isinstance(env.observation_space, spaces.Box)
    assert isinstance(env.action_space, spaces.Box)
    #Above instances just tell Pyrefly to not worry
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0]) #Instead of hard coding a value, doing it this way ensures portability

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Standard MLP TD3 for data generation
    agent = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        use_transformer=False
    )

    replay_buffer = ReplayBuffer(state_dim, action_dim, device=device)

    history = []
    episode_reward = 0
    episode_timesteps = 0
    obs, info = env.reset()
    
    total_episodes = 0
    
    print(f"Starting Algorithm Distillation Data Generation for {env_name}...")
    
    for t in range(total_timesteps):
        episode_timesteps += 1

        # Select action with exploration noise (Fills ReplayBuffer with diverse data before the AI selects on its own, and also better stability)
        if t < 25000:
            action = env.action_space.sample()
        else:
            # Noise filter: Linearly decay noise from 0.1 to 0.01 and clip to prevent harshness
            noise_scale = max(0.01, 0.1 * (1 - t / total_timesteps))
            noise = np.random.normal(0, max_action * noise_scale, size=action_dim)
            noise = noise.clip(-max_action * 0.5, max_action * 0.5) # The Filter
            
            action = (
                agent.select_action(np.array(obs))
                + noise
            ).clip(-max_action, max_action)

        # Perform action
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store transition in CURRENT episode history
        # (s, a, r) - This is what the Transformer will see in context
        history.append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'done': done
        })

        # Standard TD3 update
        replay_buffer.add(obs, action, next_obs, reward, done)
        if t >= 25000:
            agent.train(replay_buffer)

        obs = next_obs
        episode_reward += 1 # In Hopper, reward is often 1 per step + bonuses

        if done:
            total_episodes += 1
            if total_episodes % 10 == 0:
                print(f"Step: {t+1} | Episode: {total_episodes} | Reward: {episode_reward:.2f}")
            
            # Save the episode chunk every 100 episodes to prevent memory overflow
            if total_episodes % 100 == 0:
                chunk_path = os.path.join(save_dir, f"history_chunk_{total_episodes//100}.pkl")
                with open(chunk_path, 'wb') as f:
                    pickle.dump(history, f)
                history = [] # Clear memory
                print(f"Saved chunk {total_episodes//100} to {chunk_path}")

            obs, info = env.reset()
            episode_reward = 0
            episode_timesteps = 0

    # Final save (Makes sure we haven't missed anything)
    if history:
        chunk_path = os.path.join(save_dir, f"history_chunk_final.pkl")
        with open(chunk_path, 'wb') as f:
            pickle.dump(history, f)
            
    print("Data Generation Complete!")

if __name__ == "__main__":
    generate_history()
