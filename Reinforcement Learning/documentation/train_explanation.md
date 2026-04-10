# train.py: Line-by-Line Detailed Explanation

This file is the **Master Orchestrator**. it builds the world, spawns the agent, and manages the 1,000,000-step training clock.

## 🏗️ Evaluation Logic
We pause training occasionally to give the agent a \"Final Exam.\"

```python
9:  def eval_policy(policy, env_name, seed, normalizer, eval_episodes=10):
```
- **Line 9**: During evaluation, we turn off all random noise. We want to see how the agent walks when it's trying its absolute best.

---

## 🏗️ The Main Training Loop

### 🛡️ Feature 1: The \"Toddler\" Phase
```python
61:         if t < start_timesteps:
62:             action = env.action_space.sample()
```
- **Line 61-62**: For the first 25,000 steps, the agent doesn't use its brain. It just flails its legs randomly. This provides a diverse set of initial memories for the Actor to study.

### 🛡️ Feature 2: Exploration Noise
```python
68:                 + np.random.normal(0, max_action * 0.1, size=action_dim)
```
- **Line 68**: Even after learning, we add a 10% jitter to every move. This keeps the agent curious so it might accidentally discover an even better way to hop.

### 🛡️ Feature 3: Timed Out vs. Falling
```python
73:         done_bool = float(terminated) if episode_timesteps < env._max_episode_steps else 0
```
- **Line 73**: If the Hopper is still standing after 1,000 steps, the game resets. We tell the brain it's **not done** in this case because it didn't actually fail; it just ran out of time.

### 🛡️ Feature 4: Scientific Rigor (3 Seeds)
```python
101:    seeds = [0, 1, 2]
102:    for s in seeds:
103:        results = train_seed(s)
```
- **Line 101-103**: Running 3 separate seeds proves that your code works consistently and wasn't just \"getting lucky\" in one run.
