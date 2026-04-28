import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import RunningMeanStd

class RewardModel(nn.Module):
    """
    The Preference-based Reward Model (The 'Judge').
    Learns to predict a scalar reward given (state, action).
    Follows Christiano et al. (2017).
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(RewardModel, self).__init__()
        
        # We concatenate state and action as input
        input_dim = state_dim + action_dim
        
        # DEFINITIVE FIX: The Judge now has a complete brain (Sequential MLP)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
            # REPAIR LINE: We added Tanh to 'squash' the Judge's opinions into a [-1, 1] range.
            # This prevents reward explosion, which would otherwise overwhelm the agent's brain.
            nn.Tanh() 
        )
        
        # ADDED: Reward Normalizer to keep the agent stable
        self.reward_rms = RunningMeanStd()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, action):
        """Returns the predicted reward for a given state-action pair."""
        x = torch.cat([state, action], dim=-1)
        return self.mlp(x)

    def predict_reward(self, state, action, update_rms=True):
        """Predicts normalized reward for a single (s, a) pair."""
        # Convert to tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action).to(self.device).unsqueeze(0)
            
        self.eval()
        with torch.no_grad():
            r_raw = self.forward(state, action).cpu().numpy().flatten()
            
            if update_rms:
                self.reward_rms.update(r_raw)
            
            # Normalize: Subtract mean and divide by standard deviation
            # This keeps rewards centered and prevents 'Exploding Rewards'
            r_norm = (r_raw - self.reward_rms.mean) / np.sqrt(self.reward_rms.var + 1e-8) #Z-Score Normalization
            return float(r_norm[0])
            
    def get_segment_reward(self, state_seq, action_seq):
        """Calculates the sum of rewards for a segment of experience."""
        x = torch.cat([state_seq, action_seq], dim=-1) #Joins them along feature axis
        rewards = self.mlp(x)
        return torch.sum(rewards, dim=1)

class RewardEnsemble(nn.Module):
    """
    A 'Jury' of RewardModels. 
    By using multiple judges, we can prevent 'Reward Hacking' 
    (where the agent fools a single judge).
    """
    def __init__(self, state_dim, action_dim, num_models=3, hidden_dim=256):
        super(RewardEnsemble, self).__init__()
        self.models = nn.ModuleList([
            RewardModel(state_dim, action_dim, hidden_dim) for _ in range(num_models)
        ])
        
    def predict_reward(self, state, action, method="mean"):
        """
        Returns a consensus reward from the jury.
        - 'mean': Average of all judges.
        - 'min': Pessimistic estimate (Safety First).
        """
        rewards = []
        for model in self.models:
            rewards.append(model.predict_reward(state, action))
            
        if method == "mean":
            return np.mean(rewards)
        elif method == "min":
            return np.min(rewards)
        return np.mean(rewards)

    def get_segment_rewards(self, state_seq, action_seq):
        """Returns a list of segment rewards, one from each judge."""
        return [model.get_segment_reward(state_seq, action_seq) for model in self.models]
