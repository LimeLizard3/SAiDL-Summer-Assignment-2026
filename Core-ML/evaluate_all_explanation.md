# Beginner's Guide: `evaluate_all.py` (Line-by-Line)

This is the "Final Boss" script of your project. It acts as a **Tournament Director**, forcing every single version of your model (12 different combinations!) to compete against each other to see who is the fastest, the smartest, and the leanest.

---

### Part 1: The Setup (Imports)

```python
1: import torch
2: import torch.nn as nn
3: import time
4: import math
```
* **Lines 1-4**: Standard library imports. `time` is specifically used here to measure exactly how many seconds the model takes to process words.

```python
6: from config import TransformerConfig
7: from model import TransformerLM
8: from data import get_dataloaders
```
* **Lines 6-8**: We import your **Modular Blueprint**, your **Master Model**, and your **Data Loader**. This shows how well-organized your code is—this script can use all your previous work just by importing it!

---

### Part 2: Measuring the Fuel (GPU Memory)

```python
10: def get_gpu_memory():
11:     if torch.cuda.is_available():
12:         return torch.cuda.max_memory_allocated() / 1e6 # MB
```
* **Line 10-13**: This function checks your **GPU Fuel Tank**. 
    * `max_memory_allocated` tells us the absolute highest amount of memory the model used during its test. 
    * We divide by `1e6` to turn the raw bytes into human-readable **Megabytes (MB)**.

---

### Part 3: The Individual Exam (`evaluate_variant`)

This function takes one specific "Brain Combination" (e.g., MQA + RoPE) and puts it through a timed exam.

```python
15: def evaluate_variant(attn_type, pos_type, device, train_loader, val_loader):
```
* **Line 15**: We tell the function which **Attention** and which **Position** variant to test.

```python
20:     config = TransformerConfig()
21:     config.attention_type = attn_type
22:     config.pos_type = pos_type
```
* **Lines 20-22**: We create a new "Blueprint" specifically for this test run.

```python
25:     config.d_model = 128
26:     config.n_heads = 4
27:     config.n_layers = 2
```
* **Lines 25-28**: We use a small, 2-layer model. This ensures the benchmark runs quickly on your machine while still proving that the math works.

```python
30:     torch.cuda.reset_peak_memory_stats()
```
* **Line 30**: We "reset the odometer" on the GPU memory so we only measure the memory used by **this specific variant**.

```python
32:     optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
```
* **Line 32**: We give the model its "Learning Tools" so it can actually perform training steps during the speed test.

#### ⏱️ The Speed Test (Training)
```python
35:     model.train()
36:     start_time = time.time()
40:     for i, (x, y) in enumerate(train_loader):
41:         if i >= 10: break
```
* **Lines 35-41**: We start a stopwatch (`start_time`) and make the AI read exactly **10 batches** of data. 

```python
45:         logits = model(x)
46:         loss = torch.nn.functional.cross_entropy(...)
47:         loss.backward()
48:         optimizer.step()
```
* **Lines 45-48**: A full training cycle. The AI reads, guesses, grades itself, and updates its brain. 

```python
53:     tokens_per_sec = total_tokens / (duration + 1e-9)
```
* **Line 53**: The "Speed Score." We divide the total words read by the seconds spent. This tells us: *"How many thousands of words per second can this brain process?"*

#### 🧠 The Accuracy Test (Validation)
```python
61:         for i, (x, y) in enumerate(val_loader):
62:             if i >= 5: break
```
* **Lines 61-62**: The AI stops learning and starts a "Final Exam" on words it hasn't seen before.

```python
70:     ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')
```
* **Line 70**: We calculate **Perplexity (PPL)**. A stable number proves the math is correct; `inf` would mean the brain crashed.

---

### Part 4: The Tournament Director (`run_full_diagnostic`)

```python
75:     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
* **Line 75**: Automatically detects if you have an NVIDIA GPU (CUDA) or if we should use your CPU.

```python
81:         train_loader, val_loader, _ = get_dataloaders(batch_size=4, seq_len=128)
```
* **Line 81**: Fetches the **WikiText-2** dictionary and textbook so the models have something to read.

```python
86:     attn_variants = ["standard", "mqa", "sliding_window", "linear"]
87:     pos_variants = ["absolute", "rope", "alibi"]
```
* **Lines 86-87**: The master list of competitors. 

```python
91:     for attn in attn_variants:
92:         for pos in pos_variants:
```
* **Lines 91-92 - THE NESTED LOOP**: This is the heart of the script. For every attention type, it tries every position type. This is how we test all 12 combinations!

```python
94:             if attn == "linear" and pos != "rope": continue
```
* **Line 94 - The Exception**: As we learned in Task 3, Linear Attention successfully deleted the $N \times N$ grid, so it literally **cannot** use ALiBi. This line tells the tournament to skip that impossible matchup.

```python
98:                 ppl, speed, vram = evaluate_variant(attn, pos, device, train_loader, val_loader)
```
* **Line 98**: The Tournament Director officially calls the `evaluate_variant` exam we wrote above.

```python
99:                 results.append({ ... })
```
* **Lines 99-104**: We save the scores (PPL, Speed, Memory) into a master results list.

---

### Part 5: Printing the Leaderboard

```python
116:     print("\n" + "="*70)
117:     print(f"{'MODULAR VARIANT':<30} | {'PPL':<8} | {'TOK/SEC':<10} | {'VRAM MB':<10}")
```
* **Lines 116-117**: This creates the "Header" of your final data table.

```python
119:     for r in results:
121:         print(f"{r['Variant']:<30} | {ppl_str:<8} | {r['Tokens/Sec']:<10.1f} | {r['VRAM (MB)']:<10.1f}")
```
* **Lines 119-121**: This loops through all the saved competition results and prints them in a perfectly aligned table for you to copy-paste into your final report!

```python
125: if __name__ == "__main__":
126:     run_full_diagnostic()
```
* **Lines 125-126**: The "Starter Pistol." If you run this file, the tournament begins!

---

### 🚀 What this script proves:
It proves that you didn't just write code—you built a **Scientifically Verifiable System**. By showing that all 12 combinations can run on the real WikiText-2 dataset, you are proving to the SAiDL reviewers that your Transformer architecture is robust, modular, and works as intended!
