import os
import torch
import numpy as np
import gymnasium as gym #Physics Simulator Library. Provides the virtual worlds for the AI to play and move in
from model import Normalizer
from replay_buffer import ReplayBuffer
from td3 import TD3
import matplotlib.pyplot as plt

def eval_policy(policy, env_name, seed, normalizer, eval_episodes=10):
    """Evaluates the policy over a number of episodes."""
    eval_env = gym.make(env_name) #New clean Phy simulator (completely separate from the training one that we used previously, so that we can evaluate it properly)
    eval_env.reset(seed=seed + 100) #Spawns the machine into the environment

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, _ = eval_env.reset() #Reset spits out 2 vars: state and info dict. (bug logs data etc.). We pass the non-useful variable into _ Also it resets the machine when 1 loop is done
        done = False
        while not done:
            state = normalizer(state, update=False) #False just tells Normalizer to NOT update and keep the data frozen. Otherwise, the AI would literally freak out
            action = policy.select_action(np.array(state)) #policy = TD3, if we wrote TD3.select_action(), PyTorch would crash because we're telling the blueprint
                                                           #to give us the action that we need. Instead, we call it policy 
            state, reward, terminated, truncated, _ = eval_env.step(action) #.step() feeds the AI into the simulator. Terminated tells us if it died & truncated if it ran out of time
            done = terminated or truncated #Checks to see if the AI is alive and has time
            avg_reward += float(reward) 

    avg_reward /= eval_episodes
    print(f"---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print(f"---------------------------------------")
    return avg_reward

def train_seed(seed, env_name="Hopper-v5", max_timesteps=1e6, start_timesteps=25e3, batch_size=512):
    """Trains the TD3 agent for a specific seed."""
    print(f"Training Seed: {seed}")
    
    # 1. Setup Environment
    env = gym.make(env_name)
    torch.manual_seed(seed) #Sets up the seed
    np.random.seed(seed) #Tells everything to pick their Nos.
    
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)
    state_dim = env.observation_space.shape[0] #How many sensors?
    action_dim = env.action_space.shape[0] #How many actions?
    max_action = float(env.action_space.high[0])

    # Get max episode steps safely
    max_episode_steps = getattr(env, "_max_episode_steps", None)
    if max_episode_steps is None and hasattr(env, "spec") and env.spec is not None:
        max_episode_steps = env.spec.max_episode_steps
    if max_episode_steps is None:
        max_episode_steps = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Setup Components
    policy = TD3(state_dim, action_dim, max_action, device)
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    normalizer = Normalizer(shape=(state_dim,))

    if not os.path.exists("./results"): #Checks if file exists
        os.makedirs("./results") #If it doesn't makes a new directory for it
    if not os.path.exists("./models"):
        os.makedirs("./models")

    # 3. Training Loop
    state, _ = env.reset(seed=seed)
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    evaluations = []
    max_eval_reward = -float('inf')
    
    # [STABILITY] Use a safe filename to avoid overwriting your original baseline
    base_file_name = f"TD3_{env_name}_{seed}_stable"
    model_path = f"./models/{base_file_name}"
    best_model_path = f"./models/{base_file_name}_best"

    for t in range(int(max_timesteps)):
        episode_timesteps += 1

        # Select action randomly or from policy
        if t < start_timesteps:
            action = env.action_space.sample() #We're doing this to just get a varying amount of data from doing random actions till a certain frame
        else:
            state_norm = normalizer(state, update=True) #We WANT the Normalizer to learn and train now 
            
            # [STABILITY] Noise Decay: Smoothly reduce randomness as the agent masters the task
            noise_scale = max(0.02, 0.1 * (1 - t / (max_timesteps * 0.5)))
            
            action = (
                policy.select_action(np.array(state_norm))
                + np.random.normal(0, max_action * noise_scale, size=action_dim) #This line just adds some noise so that the AI could maybe discover a better way
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done_bool = float(terminated) if episode_timesteps < max_episode_steps else 0.0

        # Store in buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += float(reward)

        # Train agent
        if t >= start_timesteps:
            policy.train(replay_buffer, batch_size)

        if terminated or truncated:
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Reward: {episode_reward:.3f}")
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate periodically
        if (t + 1) % 5000 == 0:
            eval_reward = eval_policy(policy, env_name, seed, normalizer)
            evaluations.append(eval_reward)
            np.save(f"./results/{base_file_name}", evaluations)
            policy.save(model_path)
            
            # [STABILITY] Champion Memory: Save our all-time best performer separately
            if eval_reward > max_eval_reward:
                max_eval_reward = eval_reward
                print(f"New Champion! Reward: {max_eval_reward:.3f}. Saving best model...")
                policy.save(best_model_path)
                normalizer.save(best_model_path)

    return evaluations

if __name__ == "__main__":
    seeds = [0, 1, 2]
    all_results = {}
    
    for s in seeds:
        results = train_seed(s)
        all_results[s] = results

    # Final Plotting (Average over seeds)
    plt.figure(figsize=(10, 6))
    for s, res in all_results.items():
        plt.plot(res, label=f"Seed {s}")
    
    plt.title("TD3 Baseline on Hopper-v5")
    plt.xlabel("Evaluation Step (x5000)")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.savefig("td3_baseline_results_stabilized.png")
    # [HEADLESS] Disabled plt.show() to prevent blocking in automated queues
    # plt.show()
