import os
import torch
import numpy as np
import gymnasium as gym
from collections import deque
from model import Normalizer
from replay_buffer import ReplayBuffer
from td3 import TD3
from reward_model import RewardModel
from rlhf_trainer import RLHFTrainer
import matplotlib.pyplot as plt

def train_rlhf(seed=0, env_name="Hopper-v5", max_timesteps=500000, seq_len=32):
    """
    Trains a Transformer agent using RLHF (Preference-based learned rewards).
    """
    print(f"\n--- Starting RLHF Training (Seed={seed}) ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # 1. Initialize Components
    policy = TD3(state_dim, action_dim, max_action, device, use_transformer=True, seq_len=seq_len)
    
    # Champion Model Integration
    # We load the L=32 Champion weights as the seed for RLHF training.
    # This turns RLHF from a 'tabula rasa' task into a 'fine-tuning' task.
    champion_path = "./models/TD3_Transformer_L32_S0_stable_best"
    if os.path.exists(champion_path + "_actor"):
        print("Initializing RLHF Agent with L=32 Champion weights...")
        policy.actor.load_state_dict(torch.load(champion_path + "_actor", map_location=device))
        policy.actor_target.load_state_dict(policy.actor.state_dict())
        policy.critic.load_state_dict(torch.load(champion_path + "_critic", map_location=device))
        policy.critic_target.load_state_dict(policy.critic.state_dict())
    
    # REPAIR LINE: We force-load the Expert L=32 normalizer immediately.
    normalizer = Normalizer(shape=(state_dim,))
    if os.path.exists(champion_path + "_normalizer.npz"):
        print("Synchronizing Normalizer scales with L=32 Expert...")
        normalizer.load(champion_path)
    
    # Use the new high-quality Teacher Buffer for pre-training the Judge
    old_buffer_path = "./models/teacher_buffer"
    old_buffer = ReplayBuffer(state_dim, action_dim, device=device) #Getting a blank "Bookshelf"
    
    if os.path.exists(old_buffer_path + "_buffer.npz"):
        print("Loading 1M step buffer for Reward Model pre-training...")
        old_buffer.load(old_buffer_path)
    else:
        print("CRITICAL: Old buffer not found. Reward model will start untrained.")
        
    reward_model = RewardModel(state_dim, action_dim).to(device)
    rlhf_trainer = RLHFTrainer(reward_model, device=device)
    
    # Results tracking
    if not os.path.exists("./results_rlhf"):
        os.makedirs("./results_rlhf")
    results_path = f"./results_rlhf/TD3_RLHF_S{seed}.npy"
    
    # 2. Pre-train the Reward Model
    if old_buffer.size > 0:
        print("Pre-training the Judge (RLHF Reward Model)...")
        for i in range(5000):
            loss = rlhf_trainer.train_step(old_buffer, batch_size=64, segment_len=50)
            if (i+1) % 1000 == 0:
                print(f"Pre-train Iteration {i+1}/5000, Loss: {loss:.4f}")

    # 3. RL Training Loop with substitution
    new_buffer = ReplayBuffer(state_dim, action_dim, device=device)
    state, _ = env.reset(seed=seed)
    state_history = deque([np.zeros(state_dim) for _ in range(seq_len)], maxlen=seq_len)
    action_history = deque([np.zeros(action_dim) for _ in range(seq_len)], maxlen=seq_len)
    
    evaluations = []
    episode_timesteps = 0
    total_it = 0
    
    for t in range(int(max_timesteps)):
        total_it += 1
        episode_timesteps += 1
        
        # Select action using current policy
        state_norm = normalizer(state, update=True)
        state_history.append(state_norm)
        
        # Standard TD3 exploration noise
        action = (
            policy.select_action(state_norm, state_history, action_history)
            + np.random.normal(0, max_action * 0.1, size=action_dim) #0 is median, std is 10% of motor power, and size is for how many motors
        ).clip(-max_action, max_action)
        
        next_state, env_reward, terminated, truncated, _ = env.step(action)
        action_history.append(action)
        
        # --- THE RLHF SUBSTITUTION ---
        # The agent NEVER sees 'env_reward'. It only sees what the Judge thinks.
        with torch.no_grad():
            s_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(device) #Unsqueeze makes the NN think "batch of size 1"
            a_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)
            learned_reward = reward_model(s_tensor, a_tensor).item()
        
        done_bool = float(terminated) if episode_timesteps < env._max_episode_steps else 0
        new_buffer.add(state, action, next_state, learned_reward, done_bool) #The bot only looks at the scores the judge gives it
        #learned_reward replaces env_reward so that it learns from the judge
        
        state = next_state
        
        # Training steps
        if t > 5000:
            policy.train(new_buffer, batch_size=512)
            
            # REPAIR LINE: THE ETERNAL TEXTBOOK PROTOCOL
            # Every 2000 steps, we force the Judge to re-study the Expert Textbook (old_buffer)
            # and then learn from the Student's new experiences (new_buffer).
            # This prevents 'Catastrophic Forgetting' and ensures the Judge stays an expert.
            if t % 2000 == 0:
                rlhf_trainer.train_step(old_buffer, batch_size=16, segment_len=50) # Reinforce Expert Knowledge
                rlhf_trainer.train_step(new_buffer, batch_size=16, segment_len=50) # Adapt to New Student Data

        if terminated or truncated:
            state, _ = env.reset()
            episode_timesteps = 0
            state_history = deque([np.zeros(state_dim) for _ in range(seq_len)], maxlen=seq_len)
            action_history = deque([np.zeros(action_dim) for _ in range(seq_len)], maxlen=seq_len)

        # Evaluation (We evaluate against GROUND TRUTH to see if the AI learned the right thing)
        if (t + 1) % 5000 == 0:
            avg_reward = 0.
            for _ in range(10):
                s, _ = eval_env.reset()
                d = False
                s_h = deque([np.zeros(state_dim) for _ in range(seq_len)], maxlen=seq_len)
                a_h = deque([np.zeros(action_dim) for _ in range(seq_len)], maxlen=seq_len)
                while not d:
                    s_n = normalizer(s, update=False) #Preventing math from changing during exam
                    s_h.append(s_n)
                    act = policy.select_action(s_n, s_h, a_h)
                    s, r, term, trunc, _ = eval_env.step(act)
                    a_h.append(act)
                    avg_reward += r #We use the env.reward to test and see if it's working well
                    d = term or trunc
            
            avg_reward /= 10
            evaluations.append(avg_reward)
            print(f"Step {t+1}: RLHF Agent Ground-Truth Reward: {avg_reward:.2f}")
            np.save(results_path, evaluations)
            plot_rlhf_results(results_path, seed)

    print("\n[SUCCESS] RLHF Training complete. Final graph generated.")
    return evaluations

def plot_rlhf_results(rlhf_path, seed):
    """Generates the RLHF vs MLP comparison plot."""
    mlp_path = "./results/TD3_Hopper-v5_0_stable.npy"
    save_path = f"./analysis/rlhf_performance_S{seed}_stable.png"
    
    if not os.path.exists("./analysis"):
        os.makedirs("./analysis")
        
    plt.figure(figsize=(12, 7))
    
    # 1. Plot RLHF Data
    if os.path.exists(rlhf_path):
        rlhf_data = np.load(rlhf_path)
        x_rlhf = np.arange(len(rlhf_data)) * 5000
        plt.plot(x_rlhf, rlhf_data, label="Transformer (RLHF - Learned Rewards)", color='purple', linewidth=2.5)
        if len(rlhf_data) > 5:
            z = np.polyfit(x_rlhf, rlhf_data, 3)
            p = np.poly1d(z)
            plt.plot(x_rlhf, p(x_rlhf), "r--", alpha=0.5, label="RLHF Trend")
            
    # 2. Plot MLP Baseline
    if os.path.exists(mlp_path):
        mlp_data = np.load(mlp_path)
        x_mlp = np.arange(len(mlp_data)) * 5000
        plt.plot(x_mlp, mlp_data, label="Standard TD3 (MLP)", color='black', linestyle='--', alpha=0.4)

    plt.title(f"Task 2d: RLHF Ground-Truth Performance (Seed {seed})", fontsize=14, fontweight='bold')
    plt.xlabel("Training Steps")
    plt.ylabel("Reward (Environment)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Graph updated: {save_path}")

if __name__ == "__main__":
    train_rlhf(seed=0)
