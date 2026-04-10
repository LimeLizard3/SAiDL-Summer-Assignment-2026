import os
import torch
import numpy as np
import gymnasium as gym
from collections import deque
from model import Normalizer
from replay_buffer import ReplayBuffer
from td3 import TD3
import matplotlib.pyplot as plt

def eval_policy(policy, env_name, seed, normalizer, seq_len, eval_episodes=10):
    """Evaluates the policy over a number of episodes with history."""
    eval_env = gym.make(env_name)
    eval_env.reset(seed=seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, _ = eval_env.reset()
        done = False
        
        # Initialize history for the Transformer
        state_dim = eval_env.observation_space.shape[0]
        action_dim = eval_env.action_space.shape[0]
        state_history = deque([np.zeros(state_dim) for _ in range(seq_len)], maxlen=seq_len)
        action_history = deque([np.zeros(action_dim) for _ in range(seq_len)], maxlen=seq_len)
        
        while not done:
            state_norm = normalizer(state, update=False)
            state_history.append(state_norm)
            
            # Select action based on history
            action = policy.select_action(
                state_norm, 
                state_history=list(state_history), 
                action_history=list(action_history)
            )
            
            state, reward, terminated, truncated, _ = eval_env.step(action)
            action_history.append(action)
            
            done = terminated or truncated
            avg_reward += reward

    avg_reward /= eval_episodes
    print(f"---------------------------------------")
    print(f"Evaluation (L={seq_len}) over {eval_episodes} episodes: {avg_reward:.3f}")
    print(f"---------------------------------------")
    return avg_reward

def train_transformer(seq_len, seed=0, env_name="Hopper-v5", max_timesteps=1e6, start_timesteps=25e3, batch_size=256):
    """Trains the Transformer-TD3 agent for a specific history length."""
    print(f"Training Transformer L={seq_len} Seed={seed}")
    
    env = gym.make(env_name)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize components with use_transformer=True
    policy = TD3(state_dim, action_dim, max_action, device, use_transformer=True, seq_len=seq_len)
    replay_buffer = ReplayBuffer(state_dim, action_dim, device=device)
    normalizer = Normalizer(shape=(state_dim,))

    if not os.path.exists("./results_transformer"):
        os.makedirs("./results_transformer")

    state, _ = env.reset(seed=seed)
    episode_reward = 0
    episode_timesteps = 0
    evaluations = []
    
    # Track interaction history
    state_history = deque([np.zeros(state_dim) for _ in range(seq_len)], maxlen=seq_len)
    action_history = deque([np.zeros(action_dim) for _ in range(seq_len)], maxlen=seq_len)

    for t in range(int(max_timesteps)):
        episode_timesteps += 1

        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            state_norm = normalizer(state, update=True)
            state_history.append(state_norm)
            
            action = (
                policy.select_action(state_norm, list(state_history), list(action_history))
                + np.random.normal(0, max_action * 0.1, size=action_dim)
            ).clip(-max_action, max_action)

        next_state, reward, terminated, truncated, _ = env.step(action)
        action_history.append(action)
        
        done_bool = float(terminated) if episode_timesteps < env._max_episode_steps else 0
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        if t >= start_timesteps:
            policy.train(replay_buffer, batch_size)

        if terminated or truncated:
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            state_history = deque([np.zeros(state_dim) for _ in range(seq_len)], maxlen=seq_len)
            action_history = deque([np.zeros(action_dim) for _ in range(seq_len)], maxlen=seq_len)

        if (t + 1) % 5000 == 0:
            eval_reward = eval_policy(policy, env_name, seed, normalizer, seq_len)
            evaluations.append(eval_reward)
            np.save(f"./results_transformer/TD3_L{seq_len}_S{seed}", evaluations)

    return evaluations

if __name__ == "__main__":
    lengths = [4, 8, 16, 32]
    all_results = {}
    
    for l in lengths:
        results = train_transformer(seq_len=l)
        all_results[l] = results

    # Final Comparison Plot
    plt.figure(figsize=(10, 6))
    for l, res in all_results.items():
        plt.plot(res, label=f"Transformer (L={l})")
    
    # Load Baseline Seed 0 for comparison
    baseline = np.load("./results/TD3_Hopper-v5_0.npy")
    plt.plot(baseline, label="MLP Baseline", linestyle="--")
    
    plt.title("Transformer vs MLP Baseline on Hopper-v5")
    plt.xlabel("Evaluation Step (x5000)")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.savefig("transformer_comparison_results.png")
    plt.show()
