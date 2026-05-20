# pyrefly: ignore [missing-import]
import gymnasium as gym

class DelayedRewardWrapper(gym.RewardWrapper):
    """
    Wraps an environment to delay rewards until every K steps.
    Formula: r_t_delayed = sum(r_{t-K+1}...r_t) if t % K == 0 else 0
    """
    def __init__(self, env, k=10):
        super().__init__(env)
        self.k = k
        self.accumulated_reward = 0.0
        self.steps_since_reward = 0

    def reward(self, reward):
        self.accumulated_reward += reward
        self.steps_since_reward += 1
        
        if self.steps_since_reward % self.k == 0:
            delayed_r = self.accumulated_reward
            self.accumulated_reward = 0.0
            return delayed_r
        else:
            return 0.0

    def reset(self, **kwargs):
        self.accumulated_reward = 0.0
        self.steps_since_reward = 0
        return self.env.reset(**kwargs)
