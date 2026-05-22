# pyrefly: ignore [missing-import]
import os
import torch
import numpy as np
import gymnasium as gym
import argparse
from collections import deque
from model import Normalizer
from replay_buffer import ReplayBuffer
from td3 import TD3
from delayed_reward_wrapper import DelayedRewardWrapper

def process_obs(obs, mode):
    """
    Applies stressors to observations if needed.
    Under 'partial' mode (POMDP), we mask all velocity dimensions (indices 6 to 10).
    """
    if mode == "partial":
        obs_masked = obs.copy()
        obs_masked[6:] = 0.0
        return obs_masked
    return obs

def eval_policy(policy, env_name, seed, normalizer, seq_len, mode, eval_episodes=10):
    """Evaluates the policy over a number of episodes with context/history."""
    eval_env = gym.make(env_name)
    if mode == "delayed":
        # Wrap evaluation env in DelayedRewardWrapper to match training sparsity
        eval_env = DelayedRewardWrapper(eval_env, k=10)
        
    eval_env.reset(seed=seed + 100)
    
    assert isinstance(eval_env.observation_space, gym.spaces.Box)
    assert isinstance(eval_env.action_space, gym.spaces.Box)
    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, _ = eval_env.reset()
        state = process_obs(state, mode)
        done = False
        
        state_history = deque([np.zeros(state_dim) for _ in range(seq_len)], maxlen=seq_len)
        action_history = deque([np.zeros(action_dim) for _ in range(seq_len)], maxlen=seq_len)
        
        while not done:
            state_norm = normalizer(state, update=False)
            state_history.append(state_norm)
            
            # Select action
            action = policy.select_action(
                state_norm, 
                state_history=state_history, 
                action_history=action_history
            )
            
            state, reward, terminated, truncated, _ = eval_env.step(action)
            state = process_obs(state, mode)
            action_history.append(action)
            
            done = terminated or truncated
            avg_reward += float(reward)

    avg_reward /= eval_episodes
    print(f"Evaluation (L={seq_len}, Mode={mode.upper()}) over {eval_episodes} episodes: {avg_reward:.3f}")
    return avg_reward

def main():
    parser = argparse.ArgumentParser(description="Train xLSTM / Transformer agent under environmental challenges")
    parser.add_argument("--policy", type=str, choices=["xlstm", "transformer"], default="xlstm", help="Actor backbone architecture")
    parser.add_argument("--mode", type=str, choices=["clean", "partial", "delayed"], default="clean", help="Stressor mode")
    parser.add_argument("--seq_len", type=int, default=16, help="History window length")
    parser.add_argument("--max_timesteps", type=int, default=250000, help="Total training steps")
    parser.add_argument("--start_timesteps", type=int, default=25000, help="Random exploration steps")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_save_dir = os.path.join(script_dir, "results_xlstm")
    parser.add_argument("--save_dir", type=str, default=default_save_dir, help="Results save directory")
    args = parser.parse_args()

    print(f"\n========================================")
    print(f"Policy: {args.policy.upper()} | Mode: {args.mode.upper()} | Seq Len: {args.seq_len} | Seed: {args.seed}")
    print(f"========================================\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env_name = "Hopper-v5"
    env = gym.make(env_name)
    if args.mode == "delayed":
        env = DelayedRewardWrapper(env, k=10)

    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Get max episode steps safely
    max_episode_steps = getattr(env, "_max_episode_steps", None)
    if max_episode_steps is None and hasattr(env, "spec") and env.spec is not None: #If a wrapper has a spec, we can look and find the original limit defined when the env was registered (1000 for Hopper btw)
        max_episode_steps = env.spec.max_episode_steps
    if max_episode_steps is None:
        max_episode_steps = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine architecture flag
    use_xlstm = (args.policy == "xlstm")
    use_transformer = (args.policy == "transformer")

    # Initialize components
    policy = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        use_transformer=use_transformer,
        use_xlstm=use_xlstm,
        seq_len=args.seq_len
    )
    
    replay_buffer = ReplayBuffer(state_dim, action_dim, device=device)
    normalizer = Normalizer(shape=(state_dim,))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists("./models"):
        os.makedirs("./models")

    file_name = f"TD3_{args.policy}_L{args.seq_len}_M_{args.mode}_S{args.seed}"
    model_path = f"./models/{file_name}"
    best_model_path = f"./models/{file_name}_best"
    results_path = f"{args.save_dir}/{file_name}.npy"

    evaluations = []
    max_eval_reward = -float('inf') #Ensures that the first "Best model" is saved properly

    state, _ = env.reset(seed=args.seed)
    state = process_obs(state, args.mode)
    episode_reward = 0
    episode_timesteps = 0
    
    state_history = deque([np.zeros(state_dim) for _ in range(args.seq_len)], maxlen=args.seq_len)
    action_history = deque([np.zeros(action_dim) for _ in range(args.seq_len)], maxlen=args.seq_len)

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            state_norm = normalizer(state, update=True)
            state_history.append(state_norm)
            
            # Noise decay over training progression
            noise_scale = max(0.02, 0.1 * (1 - t / (args.max_timesteps * 0.5)))
            
            action = (
                policy.select_action(state_norm, state_history, action_history)
                + np.random.normal(0, max_action * noise_scale, size=action_dim)
            ).clip(-max_action, max_action)
            
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = process_obs(next_state, args.mode)
        action_history.append(action)
        
        done_bool = float(terminated) if episode_timesteps < max_episode_steps else 0.0
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += float(reward)

        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if terminated or truncated:
            state, _ = env.reset()
            state = process_obs(state, args.mode)
            episode_reward = 0
            episode_timesteps = 0
            state_history = deque([np.zeros(state_dim) for _ in range(args.seq_len)], maxlen=args.seq_len)
            action_history = deque([np.zeros(action_dim) for _ in range(args.seq_len)], maxlen=args.seq_len)

        if (t + 1) % 5000 == 0: 
            eval_reward = eval_policy(policy, env_name, args.seed, normalizer, args.seq_len, args.mode)
            evaluations.append(eval_reward)
            np.save(os.path.splitext(results_path)[0], evaluations) 
            policy.save(model_path)
            normalizer.save(model_path)
            
            if eval_reward > max_eval_reward:
                max_eval_reward = eval_reward
                print(f"New Best Performance: {max_eval_reward:.3f}. Saving best model...")
                policy.save(best_model_path)
                normalizer.save(best_model_path)

if __name__ == "__main__":
    main()
