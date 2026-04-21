# Deep Dive: `extrapolation_test.py` (Line-by-Line)

`extrapolation_test.py` is the "Final Exam" for your Transformer. It doesn't teach the AI anything; it simply tests if the AI's math stays stable when we force it to read context lengths longer than its training (512, 1024, and 2048).

---

### Part 1: Setting up the Test Lab

```python
1: import torch
2: import torch.nn as nn
3: import math
```
* **Lines 1-3**: We import our essential math tools. `torch` is the core AI library, and `math` is used for calculating Perplexity.

```python
5: from config import TransformerConfig
6: from model import TransformerLM
```
* **Lines 5-6**: We import **Your Code**. This is why your modular architecture is so great—we can easily import your custom model and settings into this separate benchmark script.

---

### Part 2: The `evaluate_length` Function (The Stress Test)

This function takes a single model and forces it to read a specific number of tokens to see if it panics.

```python
9: def evaluate_length(model, device, seq_len, batch_size=2):
```
* **Line 9**: We define the test. `seq_len` is the most important variable—it's how many words the AI has to read (e.g., 512 or 2048).

```python
14:     model.eval()
```
* **Line 14**: We put the AI in **Evaluation Mode**. This turns off things like "Dropout" so the AI stays focused and consistent.

```python
16:     torch.manual_seed(42)
```
* **Line 16**: We set a "Seed." This ensures that every time we run the test, the AI sees the **exact same random words**. If we didn't do this, one variant might get an easier "exam" than the others.

```python
20:     x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
21:     y = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
```
* **Lines 20-21**: We generate temporary "Greeble" data. 
    * `x`: The words the AI reads. 
    * `y`: The correct answers the AI should have guessed.

```python
23:     with torch.no_grad():
24:         logits = model(x)
```
* **Lines 23-24**: 
    * `no_grad()`: Tells the computer to save memory by NOT tracking how to learn. We are just testing right now. 
    * `logits`: This is the AI's raw output. It's a massive list of guesses for every word in the sequence.

```python
26:         loss = torch.nn.functional.cross_entropy(...)
```
* **Line 26**: We "Grade" the AI. This calculates how "surprised" the AI was by the correct answer.

```python
30:         ppl = math.exp(loss.item()) if loss.item() < 20 else float('inf')
```
* **Line 30 - The Survival Check**: 
    * We convert the "Loss" into **Perplexity (PPL)**. 
    * **If PPL is low**: The AI is calm and its math is stable. 
    * **If PPL is `inf` (Infinity)**: It means the AI's math literally "exploded" because the sequence was too long. This is common with "Absolute" encodings at 2048 length.

---

### Part 3: The `run_benchmark` Function (The Tournament)

This is the main "Event" that coordinates between all the different positional types.

```python
38:     lengths = [512, 1024, 2048]
39:     variants = ["absolute", "rope", "alibi"]
```
* **Lines 38-39**: We define our competitors. We will test **3 Different Brains** across **3 Different Difficulty Levels**.

```python
43:     for v in variants:
44:         print(f"\nTesting Variant: {v}")
```
* **Lines 43-44**: This loop starts the tournament. We build the "Absolute" version first, then "RoPE," then "ALiBi."

```python
45:         config = TransformerConfig()
46:         config.pos_type = v
48:         config.max_seq_len = 2048 
```
* **Lines 45-48**: We modify your blueprint (config) for the current competitor. We set the `max_seq_len` to 2048 to give it a "high ceiling."

```python
50:         model = TransformerLM(config).to(device)
```
* **Line 50**: We "Build" the model brain based on the custom config and send it to your GPU/CPU (`device`).

```python
53:         for l in lengths:
```
* **Line 53**: A nested loop! For each brain (e.g., RoPE), we test it on 512, then 1024, then 2048.

```python
54:             try:
55:                 ppl = evaluate_length(model, device, l)
```
* **Lines 54-55**: We call the `evaluate_length` stress-test we wrote earlier. 
    * We use a `try` block because if the model crashes (out of memory), we want to catch the error instead of stopping the whole tournament.

```python
62:     # Print Final Table
```
* **Line 62 onwards**: This is purely for aesthetics. It compiles all the gathered scores into a clean, professional table for your report.

---

### Part 4: The Finishing Touch

```python
73: if __name__ == "__main__":
74:     run_benchmark()
```
* **Lines 73-74**: This tells Python: *"If I run this file directly, start the tournament!"*

---

### 🚀 What this proves:

When you run this script, you are looking for **Stability**. 
*   **Absolute** (Task 1) will typically show a rising PPL as it gets closer to its memory limit. 
*   **RoPE and ALiBi** (Task 3) will stay remarkably flat and stable even at 2048, which mathematically proves that they understand "Relative" position rather than just memorizing page numbers!
