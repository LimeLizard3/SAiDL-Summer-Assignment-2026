# Beginner's Guide: `causality_test.py` (Line-by-Line)

This is the **"Lie Detector"** script of your project. Its job is to prove that your Transformer isn't cheating by peeking at future words.

---

### Part 1: The Setup

```python
1: import torch
2: import torch.nn as nn
3: from config import TransformerConfig
4: from model import TransformerLM
```
* **Lines 1-4**: Just like before, we import the core tools (`torch`) and your project’s blueprints (`config` and `model`). We need your actual model because we are going to "interrogate" it.

```python
6: def test_causality(conv_type="pre_attention"):
```
* **Line 6**: We define a function that will test one specific version of your hybrid model (either the "Pre-Attention" one or the "Interleaved" one).

```python
10:     config = TransformerConfig()
11:     config.use_conv = True
12:     config.conv_type = conv_type
13:     config.n_layers = 2
```
* **Lines 10-13**: We create a tiny, 2-layer test brain. We explicitly turn on the **Convolutional Hybrid** feature so we can put it under the microscope.

```python
15:     model = TransformerLM(config).to(device)
16:     model.eval()
```
* **Lines 15-16**: We build the model and put it in **Evaluation Mode** (`.eval()`). In this mode, the model stops learning and just sits still so we can run our experiment.

---

### Part 2: Generating the "Base" and "Modified" Sentences

This is the core of the experiment. We are going to show the model two sentences that are ALMOST identical.

```python
19:     seq_len = 10
20:     x1 = torch.randint(0, config.vocab_size, (1, seq_len)).to(device)
```
* **Line 20**: We create a random sentence of 10 words. Let’s call this **Sentence A**.

```python
24:     split_idx = 5
25:     x2 = x1.clone()
```
* **Line 25**: we create **Sentence B** as a perfect twin of Sentence A.

```python
26:     x2[0, split_idx] = (x2[0, split_idx] + 1) % config.vocab_size
```
* **Line 26 - THE CRITICAL CHANGE**: 
    * We go to the **6th word** of Sentence B and change it to something totally different.
    * **The Experiment**: Sentence A and Sentence B are identical for words 1, 2, 3, 4, and 5. Only word 6 is different.

---

### Part 3: The Interrogation

```python
28:     with torch.no_grad():
29:         output1 = model(x1)
30:         output2 = model(x2)
```
* **Lines 28-30**: We ask the AI to read both sentences. We store the way it "felt" about words 1 through 5 in two separate results (`output1` and `output2`).

---

### Part 4: The Mathematical Proof

```python
34:     diff = torch.abs(output1[0, :split_idx] - output2[0, :split_idx]).max().item()
```
* **Line 34**: We calculate the **Difference**. 
    * We compare how the model processed the first 5 words of Sentence A vs. Sentence B. 
    * **THE LOGIC**: If your model is "Causal" (it only looks at the past), it should process words 1-5 exactly the same way in both cases, because word 6 is in the **FUTURE**. 
    * If word 6 managed to change the way the model saw word 5, then your model is "Cheating" (looking at the future!).

```python
36:     if diff < 1e-5:
37:         print(f"  PASS: No future leakage detected. Max difference: {diff:.2e}")
```
* **Lines 36-37**: If the difference is practically zero (`0.00001`), it's a **PASS**! Your model obeys the "Arrow of Time."

---

### Part 5: The Starter Pistol

```python
42: if __name__ == "__main__":
43:     test_causality("pre_attention")
44:     test_causality("interleaved")
```
* **Lines 42-44**: This runs the test for both your Task 4 Hybrid designs.

### 🌟 Why this matters:
By including this script, you have moved from "Coding" to "Science." You aren’t just guessing that your CNN works; you are providing **proof** that it respects the boundaries of time, which is exactly horizontal what professional AI researchers at companies like OpenAI and Google do!
