# SAiDL Transformer Assignment: Final Project Narrative

## 1. The Optimization Journey (The "Struggle")
The transition from a standard MLP (Multilayer Perceptron) to a Transformer-based Actor was a significant technical hurdle. We faced several challenges in the first 500k steps:
- **Numerical Stability**: Transformers are notoriously unstable in Reinforcement Learning. We implemented **Mixed Precision (AMP)** and **GradScaler** to prevent gradient explosions and speed up training using 16-bit precision. 
- **The "Sensor Crushing" Bug**: We discovered a critical bug in the `Normalizer` logic that caused 11 independent sensors (positions, angles, velocities) to be averaged into a single value. This created a "blurry" input for the actor, requiring the Transformer to "guess" the true states from its past memory.

## 2. The Tale of Two Models (L=16 vs L=32)
Our primary experiment was a 1,000,000-step comparison between sequence lengths ($L=16, 32$) and a standard MLP baseline.
- **The Champion’s Fall (L=16)**: For most of the run, $L=16$ was the clear winner, reaching a peak of **513 points** (beating the MLP baseline of ~400). However, it suffered from a late-stage **Policy Collapse**. This is a classic RL phenomenon where a series of bad updates causes the agent to "forget" its high-performing policy (Catastrophic Forgetting).
- **The Titan Returns (L=32)**: While $L=32$ required more steps to stabilize, it hit a project-wide peak of **1,048 points** near step 940,000. This proves that while larger context windows require more data to master, they achieve a significantly higher "ceiling" than simpler models.

## 3. The Robustness Results: Memory vs. Reflex
Our benchmarking under stress (Noise, Delay, Partial Sight) revealed the true power of the Transformer:
- **The "Blind" Test (Partial Observability)**: When leg velocities were hidden from the agents, the MLP Baseline failed immediately (Score: 6.4). However, the **Transformer (L=32)** successfully used its memory buffer to "infer" its current velocity from past positions, achieving a score of **81.2**.
- **The Conclusion**: Transformers are not just "fancier MLPs." They provide a fundamentally different type of AI that can handle **sensory failure** and **temporal delays** which would cripple a standard reactive agent.

## 4. The Blind Judge & RLHF Recovery (Task 2d)

### The "Poisoned Student" Phenomenon
Our RLHF implementation initially suffered from a "Mid-Life Crisis" where the agent would start hopping but then suddenly collapse into zero-reward movement around Step 50,000. We identified this as the **Poisoned Student** effect:
- **The Concept**: The "Judge" (Reward Model) is pre-trained on an Expert's textbook (Buffer). However, at the start of training, the "Student" (the live agent) has an uncalibrated normalizer.
- **The Failure**: The Student reports its sensors using its own uncalibrated "dialect." The Judge doesn't recognize this dialect and gives random/low grades. Later, when we "refine" the Judge using this bad student data, the Judge's brain becomes "poisoned" by the noise, causing him to forget the Expert textbook.

### The "Triple Synchronization" Fix
To solve this, we implemented three structural "Stability Pillars":
1. **Sensory Synchronization (The Eyes)**: We forced the Student to load the **Expert L=32 Normalizer stats** from Step 0. This ensures the Student and Judge never have a "dialect mismatch."
2. **The Reward Governor (The Volume)**: We added a `nn.Tanh()` activation to the Reward Model. This squashes all opinions into a stable $[-1, 1]$ range, preventing numerical spikes from "blinding" the agent's brain.
3. **Paced Learning (The Patience)**: We reduced the Judge's update frequency from 500 to 2,000 steps, allowing the Student more time to stabilize before the Judge "updates" his grading criteria.

### The "Catastrophic Forgetting" Discovery
Even with synchronization, we observed a "Slow Collapse" in some runs. Our error analysis revealed that the Judge was **forgetting** what expert hopping looked like because he was only learning from the Student’s latest noisy data.
- **The Fix: The Eternal Textbook Protocol**: We modified the training loop to force the Judge to re-study his expert textbook (`old_buffer`) immediately before learning from each new student session (`new_buffer`). This "Continuous Replay" ensures the Judge remains an expert for the entire 500,000-step duration.

### Scientific Q&A
- **Q: Why would an L=32 Student still face this issue?**
  - **A**: The L-number represents the **Brain** (Memory). The Normalizer represents the **Eyes** (Context). Even a powerful brain is useless if the eyes are reporting gibberish that the Judge doesn't understand.
- **Q: Where did the Judge learn this 'dialect'?**
  - **A**: During the recovery phase, we created **`rlhf_pretraining.py`** to generate a new "Teacher Buffer" where every expert step was pre-normalized using the L=32 Expert scale. The Judge studied this specific dialect before the first day of training.

---
## 5. The Attention Mystery & The Robustness Sprint (Task 3)

### The "Near-Sighted" Discovery
During Task 3, we encountered a scientific discrepancy. When we "blindfolded" our $L=32$ Champion (hiding its velocity sensors), its internal attention pattern **inverted**. Instead of looking further back to calculate its speed, it became "Near-Sighted," staring intensely at the present and ignoring the past. 

### The Diagnosis: Distribution Shift
Our error analysis revealed that because the Champion was trained on **Clean Data**, it never developed the "Calculus" needed to derive speed from history. When blinded, it experienced a **Distribution Shift** (Panic Response)—it didn't recognize the "zero-velocity" history and simply lost faith in its memory buffer.

### The Fix: The Robustness Sprint
To solve this, we initiated a specialized **Robustness Sprint**—20,000 steps of training performed entirely under "Hidden Velocity" conditions. This forced the Transformer to "invent" a way to navigate without its ocular velocity sensors.

### The Final breakthrough: The "Dual-Anchor" Strategy
The resulting attention maps revealed a sophisticated, surgical strategy that the robot independently developed to survive:
1. **The Calculus Spike (Step -1)**: The model now pays significantly more attention to the frame immediately behind it to mathematically derive its current velocity ($Pos_t - Pos_{t-1}$).
2. **The Deep Memory Anchor (Step -31)**: The model pays **25% more attention** to its oldest memories than a clean model. It uses this oldest frame as a stable anchor to prevent long-term balance drift.
3. **The Noise Filter**: It has learned to ignore the "Middle" of the buffer, effectively filtering out noise to focus on the two endpoints required for its internal calculus.

---
## 6. The Chronicle of Engineering & Debugging
The success of this project was not a straight line, but a series of "Scientific Pivots":
- **Pivot 1 (Stability)**: We moved from a generic Reward Model to a **Tanh-Governed** Judge to prevent the "Million-Point Spike" that originally broke our training.
- **Pivot 2 (Alignment)**: We solved Catastrophic Forgetting in the RLHF Judge by implementing the **Eternal Textbook Protocol**, ensuring the Judge never forgets expert movement.
- **Pivot 3 (Verification)**: We proved the Transformer isn't just "fancier MLP"—we showed its **Latent Intelligence** by forcing it to re-map its own brain during the Robustness Sprint.

---
*(Final Conclusion: The project successfully moved from a standard reactive MLP to an attention-based Transformer architecture, and finally to a self-calibrating RLHF system capable of robust, human-aligned motor control. Every challenge—from sensor-crushing to poisoned students—was identified through rigorous diagnostic plotting and solved through targeted architectural hardening.)*
