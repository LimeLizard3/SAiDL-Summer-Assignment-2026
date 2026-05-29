# The Laboratory: `attention_variants.py` (Line-by-Line Breakdown)

In `model.py`, we built a standard "Brain." In this file, we are building **Experimental Modifications** for that brain. Think of these like different "Engines" you can swap into a car to see which one makes it go faster or uses less fuel.

---

### 🟢 1. The Setup (Lines 1 to 4)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
```

*   **Lines 1 to 3**: The standard PyTorch tools. `torch` for math, `nn` for LEGO-brick building blocks, and `F` for quick math functions (like Softmax).
*   **Line 4**: Standard Python `math` just so we can calculate square roots later.

---

### 🟢 2. Sliding Window Attention (Lines 6 to 52)

**The Concept**: Standard AI tries to remember *everything* in a long book at once. This is exhausting for the GPU. Sliding Window tells the AI: *"Only focus on the immediate 50 words behind you. Forget everything else."*

#### The Setup (`__init__`)
```python
class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, window_size: int = 50, dropout: float = 0.1):
        super().__init__()
```
*   **Line 7 (`__init__`)**: We pass in the dimensions, the number of heads, and the **Window Size** (default 50).
*   **Line 9 (`assert`)**: We check if the 128 decimals can be neatly divided by 4 heads. If not, we stop immediately.
*   **Lines 10-13 (`self.d_k`, etc.)**: We save our settings. `d_k` is 32 (128 / 4). We save the `window_size` (50) to a "name tag" so we can find it in the `forward` pass.

```python
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
```
*   **Lines 15-18**: We build the 4 basic "Brain Webs" (Linear Layers). Just like standard attention, we need a way to turn words into Queries (Questions), Keys (Answer Tags), and Values (Facts).

#### The Action (`forward`)
```python
    def forward(self, x, mask=None):
        bsz, seq_len, _ = x.size()
```
*   **Line 23**: Data enters the engine. We measure the Batch Size (4) and the Sequence Length (512).

```python
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
```
*   **Lines 26-28**: Just like the standard model, we project the words, slice them into 4 heads, and swap the dimensions so we can do the math Word-by-Word instead of Worker-by-Worker.

```python
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
```
*   **Line 30**: We calculate the "Raw Context Score." We multiply every Query by every Key. If they match, the score is high!

```python
        device = x.device
        window_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
```
*   **Line 33**: We check if we are running on a GPU or CPU.
*   **Line 34**: We build a standard triangular mask (all 1s in the bottom left). This prevents the AI from looking into the future.

```python
        if self.window_size > 0:
            past_mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=-self.window_size)
            window_mask = window_mask - past_mask
```
*   **Line 38**: We check if we actually want a window (e.g., 50).
*   **Line 39**: **The Magic Piece.** We create a *second* triangle mask, but we use `diagonal=-50`. This selects all the words that are *too far in the past* (older than 50 words).
*   **Line 40**: We subtract the "too old" mask from our main mask. Now, our mask is no longer a big triangle; it's a thin **Band** of 1s that "slides" along the diagonal. The AI is now effectively blinded to anything older than 50 words!

```python
        window_mask = window_mask.view(1, 1, seq_len, seq_len)
        scores = scores.masked_fill(window_mask == 0, float('-inf'))
```
*   **Line 42**: We reshape the mask into 4 dimensions so it matches the brain's shape.
*   **Line 45**: We fill all the forbidden spots (outside the window) with Negative Infinity. This ensures the AI's "Attention" on them becomes exactly 0%.

```python
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(output)
```
*   **Lines 47-52**: The final assembly. We turn scores into percentages, multiply by the Facts (Values), stitch the 4 heads back together, and send it out the door.

---

### 🟡 3. Multi-Query Attention (Lines 55 to 96)

**The Concept**: In standard AI, 4 different "Heads" keep 4 different sets of memories. MQA says: *"That's a waste of RAM. Let's have all 4 Heads share a single shared memory pool."*

#### The Setup (`__init__`)
```python
class MultiQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_k = d_model // n_heads
```
*   **Line 61**: `d_k` is 32. This is the size of one single "Head."

```python
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.d_k) 
        self.v_proj = nn.Linear(d_model, self.d_k)
```
*   **Line 64 (`q_proj`)**: Queries are still high-resolution (128 outputs). All 4 heads get their own unique questions.
*   **Lines 67-68 (`k_proj`, `v_proj`)**: **The Magic Trick.** Instead of 128 outputs, we only generate **32** outputs (exactly one head's worth). We are literally refusing to build memories for all 4 heads; we only build one memory for everyone to share. 

#### The Action (`forward`)
```python
    def forward(self, x, mask=None):
        bsz, seq_len, _ = x.size()
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
```
*   **Line 77**: We project the Queries. Notice they are still reshaped into `n_heads` (4). Every head has its own questions.

```python
        k = self.k_proj(x).view(bsz, seq_len, 1, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, 1, self.d_k).transpose(1, 2)
```
*   **Lines 80-81**: Look closely! We reshape into `1` head instead of `self.n_heads`. We are explicitly creating a "Small Memory" matrix.

```python
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
```
*   **Line 86**: **The Computational Miracle.** `q` has 4 heads, but `k` has only 1 head. PyTorch sees this and says: *"I'll just reuse that same 1 Key head for all 4 Query heads!"* This is incredibly efficient and saves massive amounts of GPU memory. This is a fantastic example of PyTorch **BROADCASTING**

---

### 🟠 4. Linear Attention (Lines 99 to 146)

**The Concept**: This is the most brilliant math trick in modern AI. Standard attention creates a massive `512 x 512` grid of focus scores. Linear Attention says: *"Wait! We can re-order the math so we never build that giant grid in the first place!"*

#### The Action (`forward`)
```python
    def forward(self, x, mask=None):
        bsz, seq_len, _ = x.size()
        q = self.q_proj(x).view(...)
        k = self.k_proj(x).view(...)
        v = self.v_proj(x).view(...)
```
*   **Lines 113-117**: Standard projection into Queries, Keys, and Values.

```python
        q = F.relu(q) + 1e-6
        k = F.relu(k) + 1e-6
```
*   **Lines 121-122**: Standard Attention uses "Softmax" to keep numbers positive and normalized. Linear Attention can't use Softmax (the math doesn't allow it). Instead, we use `ReLU` (which deletes negative numbers) and add a tiny `1e-6` so no numbers are ever zero.

```python
        kv = torch.einsum('bhtd,bhte->bhtde', k, v)
```
*   **Line 132 (`torch.einsum`) - The Memory Builder**: 
    *   **What is `einsum`?**: It stands for Einstein Summation. It’s a way to write complex matrix multiplication using "Letters" instead of trying to figure out if you should rotate or flip the matrix.
    *   **The Letters**: `b` (Batch), `h` (Heads), `t` (Time/Words), `d` (Key Dimension), and `e` (Value Dimension). (This is dim=0 to dim=4)
    *   **What's happening?**: We take the Key (`bhtd`) and the Value (`bhte`) and combine them to create a **Memory Cell** for every single word. 
    *   **The Result**: Notice the output is `bhtde`. For every word at time `t`, we now have a `32x32` square (`d` x `e`) that essentially says: *"This is what I learned from this word."*

```python
        kv_cumsum = torch.cumsum(kv, dim=2)
```
*   **Line 133 (`torch.cumsum`) - The Causal Memory**: 
    *   **Standard AI**: Uses a triangle mask to hide the future.
    *   **Linear AI**: Uses **Cumulative Sums**. By summing the memory squares over time (`dim=2`), we ensure that when the AI is at Word 50, its "Memory Cell" is the sum of Words 1, 2, 3... all the way to 50. It physically cannot see Word 51 because Word 51 hasn't been added to the sum yet! This is a legendary trick to prevent "cheating" without using a mask.
    *     Transformers don't understand past present or future, so we use time to give it a sense of ordering.

```python
        num = torch.einsum('bhtd,bhtde->bhte', q, kv_cumsum)
```
*   **Line 136 (`num`) - The Retrieval**: 
    *   We take our Query (`q`) and multiply it by our accumulated **Causal Memory** (`kv_cumsum`). This is like the AI asking a question (`q`) and looking through its accumulated "History Book" to find an answer. 

```python
        k_cumsum = torch.cumsum(k, dim=2)
        den = torch.einsum('bhtd,bhtd->bht', q, k_cumsum).unsqueeze(-1) + 1e-6
```
*   **Lines 139-140 (`den`) - Scaling the Energy**: 
    *   Because we keep adding numbers together (`cumsum`), the numbers can get huge and "explode." 
    *   We calculate a "Denominator" (`den`) by summing up all the Keys we’ve seen. This tells us exactly how much "Weight" or "Energy" we have added to the memory. 

```python
        output = num / den
```
*   **Line 142 - The Final Result**: 
    *   We divide the answer by the energy (`num / den`). This ensures that whether the book is 10 words long or 10,000 words long, the numbers in the AI's brain stay small and stable. 

```python
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(output)
```
*   **Lines 145-146**: We stitch the summary back together and send it out. **Congratulations!** You just processed the sequence without ever building a single $N \times N$ matrix!

---

### 🏛️ TASK 3 SURGERY: LINE-BY-LINE

In Task 3, we performed "Open Heart Surgery" on your three custom attention engines to make them compatible with modern positioning.

#### 1. The New "Envelope" Entrance
```python
23: def forward(self, x, mask=None, pos_params=None):
```
* **Lines 23, 83, 128**: Every variant now accepts the `pos_params` envelope. This allows the main model to pass the **Clocks** or **Penalties** down into these specialized rooms.

#### 2. The RoPE Spin (Rotary)
```python
32: if pos_params is not None and "cos" in pos_params:
33:     q = apply_rotary_emb(q, pos_params["cos"], pos_params["sin"])
34:     k = apply_rotary_emb(k, pos_params["cos"], pos_params["sin"])
```
* **Lines 32-34 (Sliding) / 94-96 (Multi-Query) / 136-138 (Linear)**: 
  * This code is identical in all three rooms. 
  * Before the Query and Key interact, we call `apply_rotary_emb`. 
  * **The Logic**: We "Spin" the word vectors in a circle. Because they are spun based on their position, the dot product (the multiplication) will naturally "feel" how far apart they are.

#### 3. The ALiBi Signal Fade (Distance Penalty)
```python
39: if pos_params is not None and "alibi_bias" in pos_params:
40:     scores = scores + pos_params["alibi_bias"][:, :, :seq_len, :seq_len]
```
* **Lines 39-40 (Sliding) / 101-102 (Multi-Query)**: 
  * We take the `scores` (the AI's interest grid) and **directly add** the negative penalty numbers.
  * **The Slice (`:seq_len`)**: Just like the "Pizza Analogy," we cut the penalty grid to match the size of the current sentence scores.
  * **Result**: Distant words get their scores crushed, making them "faded out" of the AI's hearing range.

#### 📢 The "No-Grid" Exception (Linear Attention)
Notice that **Linear Attention** (Lines 135-167) **DOES NOT** have an ALiBi section. Why?
* **The Rule**: ALiBi requires an $N \times N$ score grid to add the penalty to.
* **The Conflict**: Linear Attention's whole purpose is to **DELETE** that grid to save memory! 
* **The Solution**: Therefore, Linear Attention only uses **RoPE** for positioning, because RoPE spins individual words and doesn't need a grid to work. 

---

**Task 3 is now 100% finished, documented, and explained in every variant!** 🚀 🏗️
