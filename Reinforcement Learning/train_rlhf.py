import os
import gymnasium as gym
import numpy as np
import torch
from td3 import TD3
from model import Normalizer
from reward_model import RewardEnsemble
from rlhf_trainer import RLHFTrainer
from replay_buffer import ReplayBuffer

def train_rlhf(seed=0):
    """
    Main RLHF training loop.
    1. Loads a pre-trained Transformer (The 'Student').
    2. Loads a Jury of Reward Models (The 'Judges').
    3. The Student learns to hop by listening ONLY to the Jury.
    """
    env = gym.make("Hopper-v5")
    
    # Set seeds
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[1]
    max_action = float(env.action_space.high[0])
    seq_len = 32
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the Pre-trained Student (Transformer L=32)
    policy = TD3(state_dim, action_dim, max_action, device, use_transformer=True, seq_len=seq_len)
    champion_path = "./models/TD3_Transformer_L32_S0_stable_best"
    policy.load(champion_path)
    
    # Synchronize Normalizer
    normalizer = Normalizer(shape=(state_dim,))
    normalizer.load(champion_path)
    
    # 2. Load the Jury (RewardEnsemble)
    reward_model = RewardEnsemble(state_dim, action_dim)
    rlhf_trainer = RLHFTrainer(reward_model, device=device)
    
    # 3. Load the Expert Textbook (Teacher's Buffer)
    old_buffer = ReplayBuffer(state_dim, action_dim, device=device)
    old_buffer.load("./models/teacher_buffer")
    
    # 4. Create the Student's Learning Journal (New Buffer)
    new_buffer = ReplayBuffer(state_dim, action_dim, device=device)
    
    # 5. Pre-train the Jury on the Expert Textbook
    if not os.path.exists("rlhf_jury_expert"):
        print("Pre-training the Jury (RLHF Reward Model)... This may take a minute.")
        for i in range(5000):
            loss = rlhf_trainer.train_step(old_buffer, batch_size=64, segment_len=50)
            if (i+1) % 500 == 0:
                print(f"Pre-train Iteration {i+1}/5000, Jury Loss: {loss:.4f}")
        torch.save(reward_model.state_dict(), "rlhf_jury_expert")
    else:
        reward_model.load_state_dict(torch.load("rlhf_jury_expert"))

    # --- START TRAINING ---
    state, _ = env.reset(seed=seed)
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    
    evaluations = []
    results_path = "results_rlhf/TD3_RLHF_S0.npy"
    os.makedirs("results_rlhf", exist_ok=True)
    
    print(f"\n--- Starting RLHF Training (Seed={seed}) ---")

    for t in range(int(1e5)):
        episode_timesteps += 1

        # Select action with reduced exploration (Expert fine-tuning)
        state_norm = normalizer(state, update=False)
        action = policy.actor(
            torch.FloatTensor(state_norm).to(device).unsqueeze(0),
            torch.FloatTensor(np.zeros((1, seq_len-1, action_dim))).to(device)
        ).cpu().data.numpy().flatten()
        
        # Add tiny bit of noise for stability (5%)
        action = (action + np.random.normal(0, max_action * 0.05, size=action_dim)).clip(-max_action, max_action)

        # Perform action
        next_state, env_reward, terminated, truncated, _ = env.step(action)
        episode_reward += env_reward

        # --- THE RLHF SUBSTITUTION ---
        # The agent NEVER sees 'env_reward'. It only sees what the Judge thinks.
        learned_reward = reward_model.predict_reward(state_norm, action, method="min") * 50.0
        
        done_bool = float(terminated) if episode_timesteps < env._max_episode_steps else 0
        new_buffer.add(state, action, next_state, learned_reward, done_bool)
        
        state = next_state
        
        # Training steps
        if t > 5000:
            policy.train(new_buffer, batch_size=512)
            
            if t % 1000 == 0:
                rlhf_trainer.train_step(old_buffer, batch_size=16, segment_len=50)
                rlhf_trainer.train_step(new_buffer, batch_size=32, segment_len=50)

        if terminated or truncated:
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate periodically
        if (t + 1) % 5000 == 0:
            avg_reward = evaluate_policy(policy, "Hopper-v5", normalizer, seed)
            evaluations.append(avg_reward)
            np.save(results_path, evaluations)
            print(f"Time steps: {t+1}, Average Reward: {avg_reward:.2f}")

def evaluate_policy(policy, env_name, normalizer, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.action_space.seed(seed + 100)
    
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, _ = eval_env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            state_norm = normalizer(state, update=False)
            action = policy.actor(
                torch.FloatTensor(state_norm).to(policy.device).unsqueeze(0),
                torch.FloatTensor(np.zeros((1, 31, 3))).to(policy.device)
            ).cpu().data.numpy().flatten()
            state, reward, terminated, truncated, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    return avg_reward

if __name__ == "__main__":
    train_rlhf(seed=0)
