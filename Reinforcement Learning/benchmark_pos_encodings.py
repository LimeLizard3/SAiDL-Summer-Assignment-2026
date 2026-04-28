import os
import torch
import numpy as np
import gymnasium as gym
from collections import deque
from model import Normalizer
from replay_buffer import ReplayBuffer
from td3 import TD3
import matplotlib.pyplot as plt

def eval_policy(policy, env_name, seed, normalizer, seq_len, mode="partial", eval_episodes=2):
    """Evaluates the policy under partial observability."""
    eval_env = gym.make(env_name)
    eval_env.reset(seed=seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, _ = eval_env.reset()
        done = False
        
        state_dim = eval_env.observation_space.shape[0]
        action_dim = eval_env.action_space.shape[0]
        state_history = deque([np.zeros(state_dim) for _ in range(seq_len)], maxlen=seq_len)
        action_history = deque([np.zeros(action_dim) for _ in range(seq_len)], maxlen=seq_len)
        
        while not done:
            obs = state.copy()
            if mode == "partial":
                obs[6:] = 0 # Hide velocities
            
            state_norm = normalizer(obs, update=False)
            state_history.append(state_norm)
            
            action = policy.select_action(
                state_norm, 
                state_history=state_history, 
                action_history=action_history
            )
            
            state, reward, terminated, truncated, _ = eval_env.step(action)
            action_history.append(action)
            
            done = terminated or truncated
            avg_reward += reward

    avg_reward /= eval_episodes
    return avg_reward

def train_ablation(pos_type, seq_len=32, seed=0, env_name="Hopper-v5", max_timesteps=200000, start_timesteps=10e3, batch_size=512):
    print(f"\n--- Ablation Study: Encoding={pos_type} ---")
    
    env = gym.make(env_name)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize components with specific positional encoding
    policy = TD3(state_dim, action_dim, max_action, device, 
                 use_transformer=True, seq_len=seq_len, pos_encoding_type=pos_type)
    replay_buffer = ReplayBuffer(state_dim, action_dim, device=device)
    normalizer = Normalizer(shape=(state_dim,))

    if not os.path.exists("./results_ablation"):
        os.makedirs("./results_ablation")

    results_path = f"./results_ablation/Ablation_{pos_type}.npy"

    # Resume Logic: Check if a final checkpoint exists
    checkpoint_path = f"./models_ablation/TD3_Ablation_{pos_type}"
    start_t = 0
    if os.path.exists(checkpoint_path + "_actor"):
        policy.load(checkpoint_path)
        print(f"--- RESUMING {pos_type.upper()} FROM FINAL CHECKPOINT ---")
        if os.path.exists(results_path):
            evaluations = list(np.load(results_path)) #We need this to be a list, if it's not we can't easily add the next scores & it's not part of the movement
            start_t = len(evaluations) * 25000
        else:
            evaluations = []
    else:
        evaluations = []
        
    state, _ = env.reset(seed=seed)
    episode_reward = 0
    episode_timesteps = 0
    
    state_history = deque([np.zeros(state_dim) for _ in range(seq_len)], maxlen=seq_len)
    action_history = deque([np.zeros(action_dim) for _ in range(seq_len)], maxlen=seq_len)

    for t in range(start_t, int(max_timesteps)):
        episode_timesteps += 1

        # Partial Observability: Hide velocities before policy and normalizer see them
        obs = state.copy()
        obs[6:] = 0 

        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            state_norm = normalizer(obs, update=True)
            state_history.append(state_norm)
            
            noise_scale = max(0.02, 0.1 * (1 - t / (max_timesteps * 0.5)))
            action = (
                policy.select_action(state_norm, state_history, action_history)
                + np.random.normal(0, max_action * noise_scale, size=action_dim)
            ).clip(-max_action, max_action)
            
        next_state, reward, terminated, truncated, _ = env.step(action)
        action_history.append(action)
        
        # Buffer stores the PARTIAL observation to force the model to learn from it
        next_obs = next_state.copy()
        next_obs[6:] = 0
        
        done_bool = float(terminated) if episode_timesteps < env._max_episode_steps else 0
        replay_buffer.add(obs, action, next_obs, reward, done_bool)

        state = next_state
        episode_reward += reward

        if t >= start_timesteps:
            # 2:1 Training Ratio Hack (Train every 2 steps for a massive speed boost)
            if t % 2 == 0:
                policy.train(replay_buffer, batch_size)
            #Reducing the lr as time goes on   
            # Update schedulers regardless to keep the LR decay accurate
            policy.actor_scheduler.step()
            policy.critic_scheduler.step()

        if terminated or truncated:
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            state_history = deque([np.zeros(state_dim) for _ in range(seq_len)], maxlen=seq_len)
            action_history = deque([np.zeros(action_dim) for _ in range(seq_len)], maxlen=seq_len)

        if (t + 1) % 25000 == 0: 
            eval_reward = eval_policy(policy, env_name, seed, normalizer, seq_len, mode="partial")
            evaluations.append(eval_reward)
            print(f"Step {t+1}: Eval Reward = {eval_reward:.2f} (Speed Optimized)")
            np.save(results_path[:-4], evaluations) 
            
        # Safety Checkpoint every 100k steps
        if (t + 1) % 100000 == 0:
            if not os.path.exists("./models_ablation"):
                os.makedirs("./models_ablation")
            policy.save(f"./models_ablation/TD3_Ablation_{pos_type}_checkpoint_{t+1}")

    # Save the final brain so we don't lose it!
    if not os.path.exists("./models_ablation"):
        os.makedirs("./models_ablation")
    policy.save(f"./models_ablation/TD3_Ablation_{pos_type}")
    
    return evaluations

if __name__ == "__main__":
    encodings = ["learned", "sinusoidal", "rope"]
    all_results = {}
    
    for enc in encodings:
        results = train_ablation(pos_type=enc, max_timesteps=1000000) 
        all_results[enc] = results

    # Plot results
    plt.figure(figsize=(10, 6))
    for enc, res in all_results.items():
        plt.plot(res, label=f"Encoding: {enc.capitalize()}")
    
    plt.title("Positional Encoding Ablation (Hidden Velocity Mode)")
    plt.xlabel("Evaluation Step (x5000)")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("./analysis/pos_encoding_ablation.png")
    print("\nBenchmark complete. Plot saved to ./analysis/pos_encoding_ablation.png")
