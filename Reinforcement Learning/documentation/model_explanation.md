# model.py: Line-by-Line Detailed Explanation

This file contains the **Brain** (Neural Networks) and the **Sensory Processor** (Normalizer) of the TD3 agent.

## 🏗️ Part 1: RunningMeanStd (The Stats Tracker)
This class tracks the running average and variance of the observation data.

```python
6:  class RunningMeanStd:
7:      \"\"\"Tracks the running mean and standard deviation of data.\"\"\"
8:      def __init__(self, epsilon=1e-4, shape=()):
9:          self.mean = np.zeros(shape, 'float64')
10:         self.var = np.ones(shape, 'float64')
11:         self.count = epsilon
```
- **Line 8**: `epsilon` is a tiny \"safety number.\" We use it to avoid dividing by zero later.
- **Line 9-10**: `mean` starts at 0 and `var` starts at 1. This is a \"fresh start\" where the brain hasn't learned anything about the environment's scale yet.
- **Line 11**: `count` tracks how many observations we've seen.

---

## 🏗️ Part 1.5: update_mean_var_count_from_moments (The Incremental Math)
This implements **Welford's Algorithm** for incremental variance calculation.

```python
23: def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
24:     delta = batch_mean - mean
25:     tot_count = count + batch_count
```
- **Line 24**: `delta` calculates the difference between our old \"long-term average\" and the average of the new data.
- **Line 25**: `tot_count` combines our old history with the new history.

---

## 🏗️ Part 2: Normalizer (The Input Filter)
This class uses the stats above to scale the raw sensor data before the neural network sees it.

```python
52:         x = (x - self.rms.mean) / (np.sqrt(self.rms.var) + 1e-8)
53:         x = np.clip(x, -self.clip_limit, self.clip_limit)
54:         return x
```
- **Line 52**: **The \"Standardization\" Formula**. It shifts the data so the average is **0** and the spread is **1**. 
- **Line 53**: **The \"Filter\"**. If a sensor glitches, `np.clip` catches it and stops the brain from \"exploding\" with crazy numbers.

---

## 🏗️ Part 3: Actor (The \"Doer\")
This network decides which muscles (motors) to move in the Hopper.

```python
68:         a = F.relu(self.l1(state))
69:         a = F.relu(self.l2(a))
70:         return self.max_action * torch.tanh(self.l3(a))
```
- **Line 68-69**: `relu` allows the brain to learn complex \"IF/THEN\" patterns.
- **Line 70**: **The \"Tanh Speed Limiter\"**. `tanh` forces the action to be between -1.0 and 1.0, ensuring the motors aren't pushed beyond their physical limits.

---

## 🏗️ Part 4: Critic (The \"Judge\")
This network judges if the Actor's moves were actually smart or not.

```python
86:         sa = torch.cat([state, action], 1)
```
- **Line 86**: `torch.cat` glues the scene (**State**) and the move (**Action**) together so the Judge has the full context.

```python
93:         q2 = F.relu(self.l4(sa))
94:         q2 = F.relu(self.l5(q2))
95:         q2 = self.l6(q2)
```
- **Line 93-95**: This is the second Judge (The Twin). By having two judges, we can pick the most pessimistic estimate, which prevents the agent from being reckless.
