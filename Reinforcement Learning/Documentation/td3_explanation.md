# td3.py: Line-by-Line Detailed Explanation

This file contains the core **TD3 (Twin Delayed Deep Deterministic Policy Gradient)** algorithm logic.

## 🏗️ Initialization
The agent holds both the primary networks and their \"targets\" (ghost copies).

```python
9:          self.actor = Actor(state_dim, action_dim, max_action).to(device)
10:         self.actor_target = copy.deepcopy(self.actor)
```
- **Line 10**: We make a \"ghost\" copy of the actor. This target network moves slowly and provides a stable target for our math.

## 🏗️ The Training Heart
This is where the agent studies its past memories to improve its judgment.

### 🛡️ Feature 1: Target Policy Smoothing
```python
49:             noise = (torch.randn_like(action) * self.policy_noise).clamp(...)
50:             next_action = (self.actor_target(next_state) + noise).clamp(...)
```
- **Line 49-50**: We intentionally add noise to the next action. This forces the agent to learn a robust strategy that doesn't break if its foot slips.

### 🛡️ Feature 2: Clipped Double-Q
```python
53:             target_Q1, target_Q2 = self.critic_target(next_state, next_action)
54:             target_Q = torch.min(target_Q1, target_Q2)
```
- **Line 53-54**: We ask both Target Critics for their score and take the **minimum**. This prevents the agent from being over-confident.

### 🛡️ Feature 3: The Bellman Equation
```python
55:             target_Q = reward + not_done * self.gamma * target_Q
```
- **Line 55**: The most important math in RL. It calculates the value of the current moment based on the reward received PLUS the discounted value of the future.

### 🛡️ Feature 4: Delayed Policy Updates
```python
65:         if self.total_it % self.policy_freq == 0:
66:             actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
```
- **Line 65**: We only update the Actor network half as often as the Critics. This lets the Judge (Critic) stabilize before the Actor makes changes.

### 🛡️ Feature 5: Soft Updates
```python
75:             target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```
- **Line 75**: Instead of a full copy, we \"blend\" a tiny bit (0.5%) of the new brain into the target brain every step.
