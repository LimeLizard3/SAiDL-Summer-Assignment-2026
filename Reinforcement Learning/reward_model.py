import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, state, action):
        """Returns the predicted reward for a given state-action pair."""
        x = torch.cat([state, action], dim=-1)
        return self.mlp(x)

    def predict_reward(self, state, action):
        """Returns scalar reward (numpy/float) for use in the environment."""
        with torch.no_grad():
            r = self.forward(state, action)
            return r.cpu().item()
            
    def get_segment_reward(self, state_seq, action_seq):
        """Calculates the sum of rewards for a segment of experience."""
        x = torch.cat([state_seq, action_seq], dim=-1)
        rewards = self.mlp(x)
        return torch.sum(rewards, dim=1)
