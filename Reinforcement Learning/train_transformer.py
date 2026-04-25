import os
import torch
import numpy as np
import gymnasium as gym
from collections import deque #It's kind of like sliding window memory but for python lists
#A special python list with a fixed maxlen. If you push a new item into a full deque, it auto kicks/pops the oldest item out the back
from model import Normalizer
from replay_buffer import ReplayBuffer
from td3 import TD3
import matplotlib.pyplot as plt

def eval_policy(policy, env_name, seed, normalizer, seq_len, eval_episodes=10):
    """Evaluates the policy over a number of episodes with history."""
    eval_env = gym.make(env_name) #Testing env. NOT the training env
    eval_env.reset(seed=seed + 100) #+100 to ensure we don't just memorize the old seed answers

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, _ = eval_env.reset() #Reset the physical simulator back to the start of the game
        done = False
        
        # Initialize history for the Transformer
        state_dim = eval_env.observation_space.shape[0] #How many sensors?
        action_dim = eval_env.action_space.shape[0] #How many actions?
        state_history = deque([np.zeros(state_dim) for _ in range(seq_len)], maxlen=seq_len)
        action_history = deque([np.zeros(action_dim) for _ in range(seq_len)], maxlen=seq_len)
        #no/zerios() creates a dummy frame and maxlen=seq_len basc says it can have no more than the max_len
        
        while not done:
            state_norm = normalizer(state, update=False) #HOLDUP! Why do we pass this through normalizer if we're not updating the values?
            #Answer: We still need to scale the values. We just don't want to update the mean/std of the normalizer because we're not training the model here.
            state_history.append(state_norm)
            
            # Select action based on history
            action = policy.select_action(
                state_norm, 
                state_history=list(state_history), 
                action_history=list(action_history) #PyTorch doesn't know how to turn a deque into a tensor, so we convert to a list
            )
            #AI's action is absolute here
            state, reward, terminated, truncated, _ = eval_env.step(action)
            action_history.append(action)
            
            done = terminated or truncated
            avg_reward += reward

    avg_reward /= eval_episodes
    print(f"---------------------------------------")
    print(f"Evaluation (L={seq_len}) over {eval_episodes} episodes: {avg_reward:.3f}")
    print(f"---------------------------------------")
    return avg_reward

def train_transformer(seq_len, seed=0, env_name="Hopper-v5", max_timesteps=1000000, start_timesteps=25e3, batch_size=512):
    """Trains the Transformer-TD3 agent for a specific history length."""
    print(f"\n--- Training Transformer L={seq_len} Seed={seed} ---")
    
    env = gym.make(env_name) #Training env
    torch.manual_seed(seed) #Ensures randomness is controlled and makes debugging possible
    np.random.seed(seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize components
    policy = TD3(state_dim, action_dim, max_action, device, use_transformer=True, seq_len=seq_len)
    replay_buffer = ReplayBuffer(state_dim, action_dim, device=device)
    normalizer = Normalizer(shape=(state_dim,))

    if not os.path.exists("./results_transformer"):
        os.makedirs("./results_transformer")
    if not os.path.exists("./models"):
        os.makedirs("./models")

    file_name = f"TD3_Transformer_L{seq_len}_S{seed}_stable"
    model_path = f"./models/{file_name}"
    best_model_path = f"./models/{file_name}_best"
    results_path = f"./results_transformer/{file_name}.npy"

    evaluations = []
    max_eval_reward = -float('inf')
    start_t = 0

    # RESUME LOGIC
    if os.path.exists(results_path) and os.path.exists(model_path + "_actor"):
        print(f"Found existing checkpoint for L={seq_len}. Loading states...")
        evaluations = list(np.load(results_path))
        start_t = len(evaluations) * 5000 # Since we evaluate every 5k steps
        
        policy.load(model_path)
        normalizer.load(model_path)
        
        if os.path.exists(model_path + "_buffer.npz"):
            print("Loading Replay Buffer...")
            replay_buffer.load(model_path)
        
        start_timesteps = 0 # No need for random exploration if we have a trained model
        print(f"Resuming from timestep {start_t} (Target: {max_timesteps})")
    else:
        print("Starting fresh training...")

    state, _ = env.reset(seed=seed)
    episode_reward = 0
    episode_timesteps = 0
    
    # Track interaction history
    state_history = deque([np.zeros(state_dim) for _ in range(seq_len)], maxlen=seq_len)
    action_history = deque([np.zeros(action_dim) for _ in range(seq_len)], maxlen=seq_len)

    for t in range(int(start_t), int(max_timesteps)):
        episode_timesteps += 1

        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            state_norm = normalizer(state, update=True)
            state_history.append(state_norm)
            
            # [STABILITY] Noise Decay: Reduce noise as training progresses
            noise_scale = max(0.02, 0.1 * (1 - t / (max_timesteps * 0.5)))
            
            action = (
                policy.select_action(state_norm, state_history, action_history)
                + np.random.normal(0, max_action * noise_scale, size=action_dim)
            ).clip(-max_action, max_action)
            
        next_state, reward, terminated, truncated, _ = env.step(action)
        action_history.append(action)
        
        done_bool = float(terminated) if episode_timesteps < env._max_episode_steps else 0
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        if t >= start_timesteps:
            policy.train(replay_buffer, batch_size)
            # [STABILITY] Slow learning rate decay for the Transformer
            policy.actor_scheduler.step()
            policy.critic_scheduler.step()

        if terminated or truncated:
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            state_history = deque([np.zeros(state_dim) for _ in range(seq_len)], maxlen=seq_len)
            action_history = deque([np.zeros(action_dim) for _ in range(seq_len)], maxlen=seq_len)

        if (t + 1) % 5000 == 0: 
            eval_reward = eval_policy(policy, env_name, seed, normalizer, seq_len)
            evaluations.append(eval_reward)
            np.save(results_path[:-4], evaluations) 
            policy.save(model_path)
            normalizer.save(model_path)
            
            # [STABILITY] Champion Memory: Save our all-time best performer separately
            if eval_reward > max_eval_reward:
                max_eval_reward = eval_reward
                print(f"New Champion! Reward: {max_eval_reward:.3f}. Saving best model...")
                policy.save(best_model_path)
                normalizer.save(best_model_path)
            # Replay buffer is large, save less frequently or only on successful exit?
            if (t + 1) % 50000 == 0:
                print("Saving persistent Replay Buffer (Large File)...")
                replay_buffer.save(model_path)

    return evaluations

if __name__ == "__main__":
    lengths = [4, 8, 16, 32]
    all_results = {}
    
    for l in lengths:
        results = train_transformer(seq_len=l) #Does having a longer length make it better?
        all_results[l] = results

    # Final Comparison Plot
    plt.figure(figsize=(10, 6))
    for l, res in all_results.items():
        plt.plot(res, label=f"Transformer (L={l})")
    
    # Load Stabilized MLP Baseline for a fair scientific comparison
    baseline = np.load("./results/TD3_Hopper-v5_0_stable.npy")
    plt.plot(baseline, label="MLP Baseline (Stabilized)", linestyle="--", color='grey', alpha=0.8)
    
    plt.title("Transformer vs MLP Baseline on Hopper-v5")
    plt.xlabel("Evaluation Step (x5000)")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.savefig("./analysis/transformer_comparison_stabilized.png")
    # [HEADLESS] Disabled plt.show() to prevent blocking in automated queues
    # plt.show()
