# The Science of GPS for AI: `positional_logic.py` (Deep Dive)

When an AI reads a sentence, it initially sees it as a "bag of words" with no order. **Task 3** gives the AI a compass and a ruler. This file, `positional_logic.py`, contains the actual mathematical "gears" that turn raw numbers into a sense of space and time.

---

### Section 1: RoPE (Rotary Positional Embeddings)

RoPE is the "Clock Face" method. It encodes position by "spinning" the word's data in a circle.

```python
5: def apply_rotary_emb(x, cos, sin):
```
*   **Line 5**: This defines our function. It takes three things:
    *   `x`: The actual word data (The "Query" or "Key").
    *   `cos` and `sin`: The pre-calculated "Clock Angles" for each position in the sentence.

```python
12:     d_k = x.size(-1)
```
*   **Line 12**: We look at the very last dimension of our data (`d_k`), which is `32` in our model. This tells us how many decimal numbers represent a single word's concept in one "Head."

```python
15:     x_even = x[..., 0::2]
16:     x_odd = x[..., 1::2]
```
*   **Lines 15-16 - The Pairing Step**: To "Spin" something in 2D, you need two coordinates (an X and a Y). 
    *   **Line 15**: We grab all the even-numbered decimals (Decimal 0, 2, 4...).
    *   **Line 16**: We grab all the odd-numbered decimals (Decimal 1, 3, 5...).
    *   **Understanding**: By pairing Decimal 0 with Decimal 1, and Decimal 2 with Decimal 3, we turn our 32-decimal list into **16 pairs**. Each pair is treated like a single point on a 2D map.

```python
23:     cos = cos[:x.size(2), :d_k//2].view(1, 1, x.size(2), d_k//2)
24:     sin = sin[:x.size(2), :d_k//2].view(1, 1, x.size(2), d_k//2)
```
*   **Lines 23-24 - The Shape Matcher**:
    1.  We slice the `cos` and `sin` tables to match the current length of the text (`x.size(2)`).
    2.  We use `.view(1, 1, ...)` to add "fake" dimensions. This is required so PyTorch can mathematically "overlay" the clock angles precisely onto the word data without crashing.

```python
26:     out = torch.empty_like(x)
```
*   **Line 26**: We create a new, empty container exactly the same size as `x` to hold our final "Spun" result.

```python
27:     out[..., 0::2] = x_even * cos - x_odd * sin
28:     out[..., 1::2] = x_even * sin + x_odd * cos
```
*   **Lines 27-28 - THE SPIN (Trigonometry)**: This is the core magic! 
    *   This is the standard formula for rotating a point $(x, y)$ by an angle $\theta$. 
    *   **The Logic**: If Word 1 is at angle 10° and Word 2 is at angle 30°, the dot product (multiplication) between them will naturally result in 20° (the distance between them). 
    *   The model "feels" the distance because of how warped the decimals became after the spin!

---

### Section 2: `get_alibi_slope` (The Hearing Range)

ALiBi doesn't use clocks; it uses "Fading Strength." This section calculates exactly how fast the signal should "fade" for each of your 4 Attention workers.

```python
35:     def get_slopes_power_of_2(n):
36:         start = (2 ** (-8 / n))
```
*   **Lines 35-36 - The "Magic 8" Algorithm**:
    *   **Start**: We calculate a "Starting Base." For 4 heads, `2 ** (-8/4)` becomes `2^-2 = 0.25`.
    *   **Why 8?**: This was the "Golden Number" discovered by researchers. It ensures the AI doesn't forget too quickly but isn't too lazy either.

```python
37:         ratio = start
38:         return [start * (ratio**i) for i in range(n)]
```
*   **Lines 37-38 - The Volume Knob**:
    *   This creates a **Geometric Progression**.
    *   **Head 1**: `0.25` (Strongest signal - can hear long distance).
    *   **Head 2**: `0.0625` (Medium signal).
    *   **Head 3**: `0.0156` (Weak signal).
    *   **Head 4**: `0.0039` (Whisper signal - only hears words next to it).
    *   **The Philosophy**: By giving our 4 heads different "Slopes," we ensure one worker focuses on the big picture (Head 1) while another focuses on local grammar (Head 4).

```python
42:     else: # The Interpolation Case
44:         closest_power_of_2 = 2**math.floor(math.log2(n_heads))
45:         slopes_base = get_slopes_power_of_2(closest_power_of_2)
46:         slopes_extra = get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:n_heads - closest_power_of_2]
```
*   **Lines 42-46 - Solving for Non-Powers of 2**: 
    *   If you had **6 heads**, you can't neatly divide the "Magic 8" formula.
    *   **The Solution**: We find the closest power of 2 below it (4), get those slopes, then calculate slopes for twice as many heads (8) and "cherry-pick" the extras to fill the gaps. This ensures a smooth mathematical gradient no matter how many heads you have.

---

### Section 3: `build_alibi_bias` (The Infinite Ruler)

This is where we actually construct the "Fading Signal" penalty grid.

```python
53:     slopes = torch.tensor(get_alibi_slope(n_heads), ...).view(n_heads, 1, 1)
```
*   **Line 53**: We turn our decimal slopes into a PyTorch "Tensor" (a grid object). The `.view(n_heads, 1, 1)` is like turning a list of numbers into a stack of layers so they can be "broadcast" across the whole sentence efficiently.

```python
60:     m = torch.arange(seq_len, device=device)
```
*   **Line 60**: We create a simple index list: `[0, 1, 2, 3...]`. This represents the position of every word.

```python
61:     distance_matrix = (m.unsqueeze(0) - m.unsqueeze(1))
```
*   **Line 61 - The 2D Ruler Trick**: 
    *   We take our 1D list and "duplicate" it horizontally and vertically.
    *   By subtracting them, we get a **Subtractive Grid**. 
    *   **Visualization**:

| | **Word 0** | **Word 1** | **Word 2** | **Word 3** |
|---|---|---|---|---|
| **Word 0** | 0 | 1 | 2 | 3 |
| **Word 1** | -1 | 0 | 1 | 2 |
| **Word 2** | -2 | -1 | 0 | 1 |
| **Word 3** | -3 | -2 | -1 | 0 |

*   Look at **Word 3** (Row) looking at **Word 0** (Column). The value is **-3**. The AI now instantly knows Word 0 is 3 steps in the past!

```python
66:     bias = slopes * distance_matrix 
```
*   **Line 66 - Applying the Penalty**: 
    *   We multiply the **Slope** (e.g., 0.25) by the **Distance** (e.g., -3).
    *   Penalty = `-0.75`.
    *   This `-0.75` is added to the word's "Attention Score." 
    *   If a word is 100 steps away, the penalty might be `-25.0`. Since attention scores are usually around 10, a score of `-15` (10 - 25) effectively makes that word **Invisible** to the AI!

```python
67:     return bias.unsqueeze(0) # (1, n_heads, seq_len, seq_len)
```
*   **Line 67**: We add one more "fake" dimension for the Batch Size (making it 4D) and send it back to the model.

---

### 🚀 Why is this better than Task 1?

In Task 1, positions were **Learned**. If the AI never saw word position #1000 during training, it wouldn't know what to do.

In Task 3, because we are using **Calculated Circles (RoPE)** and **Calculated Signal Fading (ALiBi)**, the math stays consistent forever. An AI that understands "10 steps away" on Page 1 will understand "10 steps away" on Page 10,000!

---

### 🧠 The Ghost in the Machine: Why there are no For-Loops

You might be wondering: *"Wait, we have 4 different heads (slopes). Does the AI do a for-loop 4 times?"*

The answer is: **No.** In AI, for-loops are the enemy of speed. Instead, we use a concept called **Broadcasting** (also known as Vectorization). 

#### 1. The Traditional Way (Slow)
In a normal program, you would tell the computer: *"Take Head 1, calculate its penalty. Now take Head 2, calculate its penalty..."* This is like a single messenger running back and forth 4 times. 

#### 2. The AI Way (Broadcasting - Fast)
PyTorch is smarter. When you multiply a **Vertical Stack** of 4 slopes (`4, 1, 1`) by a **Flat Sheet** of distances (`512, 512`), PyTorch instantly "beams" the distances across all 4 head-slots simultaneously. 

*   **Result**: The GPU calculates all 4 heads at the **exact same nanosecond**. 


