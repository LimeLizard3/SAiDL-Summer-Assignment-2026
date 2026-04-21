# Beginner's Guide to `model.py` (The Brain)

This file is the literal Artificial Intelligence. If `config.py` was the blueprint and `data.py` was the Librarian handing out flashcards, `model.py` is the mathematical brain that actually looks at those flashcards and attempts to learn English. 

We will break down all 115 lines continuously.

---
#### 1. The Math Toolbox (Lines 1 to 5)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from attention_variants import SlidingWindowAttention, MultiQueryAttention, LinearAttention
```
* **Lines 1 to 4**: We import `torch` (PyTorch) and its Neural Network module (`nn`). Think of `nn` as a box of pre-built LEGO bricks for AI. Instead of calculating deep calculus by hand, we just snap these LEGO bricks together.
* **Line 5**: **[Task 2 Change]** This is where we bring in the "Supercharged" attention variants we built in the separate file. It allows the model to swap its "brain" for faster alternatives like Multi-Query or Linear attention.

---
### 2. The GPS System (Positional Encoding)
If I give you the words ["Dog", "Bites", "Man"] and ["Man", "Bites", "Dog"], the exact same words mean two completely different things because of their **order**. Because Transformers read all 512 words simultaneously in a massive flash, they are legally blind to word order. 

This class manually stamps a unique mathematical "barcode" (using sine waves) onto every single word so the AI knows its exact position in the sentence.

---
### 3. The Core Concept (Standard Attention)
**[Task 2 Change]** This is the original way Transformers "think." In Task 2, we renamed this to `StandardAttention` so we could easily distinguish it from our newer, more experimental variants.

---
### 4. The Transformer Block (The Floor of the Building)
This block combines the Attention mechanism from above with the "private scratchpad" memory we discussed earlier (`d_ff`).

```python
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Select the attention engine based on config
        if config.attention_type == "mqa":
            self.attn = MultiQueryAttention(config.d_model, config.n_heads, config.dropout)
        elif config.attention_type == "linear":
            self.attn = LinearAttention(config.d_model, config.n_heads, config.dropout)
        elif config.attention_type == "sliding_window":
            self.attn = SlidingWindowAttention(config.d_model, config.n_heads, config.window_size, config.dropout)
        else: # Default is "standard"
            self.attn = StandardAttention(config.d_model, config.n_heads, config.dropout)
```
* **Lines 55 to 65 (The Switch)**: **[Task 2 Change]** This is the main surgical change we made. Instead of always using the same attention logic, the `TransformerBlock` now looks at your `config`. If you set it to `"mqa"` on your computer, the model dynamically builds a Multi-Query engine!

---
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 1024):
        super().__init__()
```
* **Lines 6 to 8**: Here we are using fundamental Python concepts to build a custom AI part.
  * **Line 6 (`class`)**: A `class` in Python is a Blueprint. Here, we are creating a blueprint for a brand-new LEGO brick called `PositionalEncoding`. The `(nn.Module)` part tells Python: *"I want my blueprint to be a sub-category of PyTorch's official Neural Network LEGO bricks so it can snap together with other AI parts."*
  * **Line 7 (`def __init__`)**: The word `__init__` stands for **Initialize**. This is the Setup Phase. Every time you build a brick from this blueprint, it runs this setup code exactly once. We use the setup phase to define its size parameters from our `config.py` (like `d_model = 128` and `max_seq_len = 1024`).
  * **Line 8 (`super().__init__()`)**: This is the most important OOP (Object-Oriented Programming) rule! Because our custom brick is piggybacking off PyTorch's official `nn.Module` blueprint, we have to courteously tell the "Super" parent blueprint to run *its* setup process first! This ensures all the underlying PyTorch math connectors are fully built before we try to add our custom barcode logic.

```python
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
```
* **Line 9**: This line physically labels the words (0, 1, 2, up to 1023). 
  * `torch.arange(max_seq_len)` creates a perfectly flat, horizontal list `[0, 1, 2, 3...]`.
  * **Wait, what is `.unsqueeze(1)`?** Neural networks require data to be structured cleanly in strict columns and grids, not just loose flat lists. The `unsqueeze(1)` command tells PyTorch: *"Take this flat measuring tape and forcefully squeeze open a 2nd dimension at index 1 to puff it up!"* It instantly snaps the flat list into a strict vertical column (1024 rows by 1 column). We do this so it perfectly aligns with the mathematical Sine Waves we multiply it by on Line 12!
* **Line 10**: Calculates a shrinking mathematical frequency (`div_term`). It uses complex exponential math so that the barcode doesn't accidentally overlap with word meanings.

```python
        pe = torch.zeros(max_seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
```
* **Line 11**: We create a completely blank canvas (`pe`) filled with zeros. Its size is exactly 1024 words long, and 128 decimals deep.
* **Lines 12 & 13**: Now we pour our 64 mathematically calculated wave speeds into the 128 blank slots!
  * **Line 12**: `0::2` means *"Start at index 0, and give me every Even numbered slot."* This generates exactly 64 empty slots in our array. We pour all 64 of our **Sine Waves** perfectly into those even slots.
  * **Line 13**: `1::2` means *"Start at index 1, and give me every Odd numbered slot."* This generates the other 64 empty slots. We reuse those exact same wave speeds to pour 64 **Cosine Waves** into the odd slots.
  * **Result**: **64 Sine Waves + 64 Cosine Waves = 128 perfectly filled slots!** Every single word now has a unique math barcode of 128 numbers identifying exactly where it sits in the sentence.

```python
        self.register_buffer('pe', pe)
```
* **Line 14**: `register_buffer` tells PyTorch: *"Save this barcode canvas into memory, but do NOT try to learn or change it during training. It is a permanent mathematical law."*

```python
    def forward(self, x):
        seq_len = x.size(1)
        pos = self.pe[:seq_len].transpose(0, 1)
        return x + pos
```
* **Lines 16 to 23**: This introduces the most important function in PyTorch: `forward`.
  * **What is a `forward` function?** While `__init__` was the "Setup Phase", `forward` is the "Action Phase." You can think of it as the physical tube that data flows through. Whenever we hand an English flashcard `x` to this specific LEGO block, PyTorch automatically runs this `forward` function to process it.
  * **Line 20**: The flashcard `x` enters the tube. The block immediately measures exactly how many words are on the flashcard using `x.size(1)`. We store that number as `seq_len` (Sequence Length).
  * **Line 22**: We don't always need all 1,024 barcodes! If the flashcard is only 50 words long, `self.pe[:seq_len]` beautifully slices out exactly the first 50 barcodes from our saved memory canvas. 
    * **Wait, why do we need `transpose(0, 1)`?** This is all about Matrix Shapes. When we originally created `self.pe`, its shape was `(1024, 1, 128)` (Words, Batch Size, Decimal Meaning). However, our incoming flashcard `x` was handed to us with a completely different shape: `x = (4, 50, 128)` (Batch Size, Words, Decimal Meaning). 
    * If we try to mathematically add `(50, 1, 128)` directly to `(4, 50, 128)`, PyTorch will instantly crash because the grid lines don't match up!
    * `transpose(0, 1)` specifically tells PyTorch to swap the 0th dimension and the 1st dimension. It takes our barcode chunk `(50, 1, 128)` and magically flips it into `(1, 50, 128)`. Now the dimensions perfectly match `x`! Because the first dimension is now `1`, PyTorch uses a built-in trick called "Broadcasting" to automatically duplicate that identical barcode sequence across all `4` items in the batch simultaneously!
  * **Line 23 (`x + pos`)**: We perform what is called "Element-wise Addition". The AI possesses 128 decimals that represent the sheer concept of the word "Apple". We take our 128 Sine/Cosine barcode decimals and literally add them directly on top of the "Apple" decimals (`number + number`). By permanently blending the two numbers together, the final math output now contains both the meaning of the word *and* its physical location in the sentence!

---
### 3. The Core Concept (Self-Attention)
This is the magic that makes AI understand context. When reading the sentence "The bank of the river", the Attention mechanism looks at the word "bank", sweeps its eyes across the rest of the sentence, sees "river", and realizes "bank" means dirt, not money.

```python
class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
```
* **Lines 25 to 28**: We create the Attention block. We pass in `d_model` (128) and `n_heads` (4 workers). Line 28 uses `assert` to verify that 128 divides perfectly by 4, otherwise the math will crash later.

```python
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
```
* **Lines 29 to 31**: This defines how the 4 workers will split the workload. `d_k` calculates exactly how much of the 128-decimal meaning each worker is allowed to look at. Since 128 divided by 4 is exactly 32, each worker gets a 32-decimal slice (`d_k`) of the word to process.
  * **Wait, why do we put `self.` in front of everything?** This is another core Python OOP trick. Remember, this block of code is inside `__init__` (the temporary Setup Phase). Any variable you create inside `__init__` is normally thrown in the garbage the exact second setup finishes! To prevent that, you attach `self.` to the front of a variable. This mathematically glues the variables to the physical LEGO brick so that months from now, if the `forward` function needs to know how many workers there are, it can literally look at the brick's name tag (`self.n_heads`) and read it! 

```python
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
```
* **Lines 33 to 38**: These are the physical neural network "brains" (`nn.Linear`) that are permanently glued to this exact LEGO block. Because they are glued with `self.`, they will remember what they learn forever. 
  * **What are the 2 parameters in `nn.Linear(128, 128)`?** Yes, they are exactly the number of Neurons! The first number is the **Input Neurons** (how many wires are coming in). The second number is the **Output Neurons** (how many wires are going out). Since we set both to 128 (`d_model`), this creates a massive web where every single one of the 128 input neurons physically connects to all 128 output neurons, creating exactly 16,384 permanent connecting synapses (`128 * 128`)!
  * We create exactly 4 of these massive `16,384`-synapse brain webs to teach the AI how words relate:
    * **`q_proj` (Query)**: This brain strictly learns to ask questions. (e.g., The word "Bank" sends a Query asking, *"Is there a word around here related to finance or rivers?"*)
    * **`k_proj` (Key)**: This brain strictly learns to answer questions! It creates name tags. (e.g., The word "River" holds up a Key that shouts, *"I relate to water and nature!"*)
    * **`v_proj` (Value)**: Holds the raw mathematical fact of what the word actually is.
    * **`out_proj` (Output)**: Once the Query and Key match up, this network acts as the final manager to cleanly format the answer.
  * **Line 38 (`self.dropout`)**: As discussed earlier, this adds 10% blindness so the 4 brain-webs don't just memorize the test answers.

```python
    def forward(self, x, mask=None):
        bsz, seq_len, _ = x.size()
```
* **Lines 40 & 41**: Data enters the Attention block. We measure how many flashcards are in the batch (`bsz` = 4) and how long they are (`seq_len` = 1024).

```python
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
```
* **Lines 43 to 45**: **Generating the Q, K, and V Math Arrays (The Multi-Head Split)**
  * **Generating Queries, Keys, and Values**: First, the incoming flashcard data (`x`) flows directly through three separate Neural Network brains that we set up earlier: `self.q_proj(x)`, `self.k_proj(x)`, and `self.v_proj(x)`. 
    * These matrices mathematically convert the raw "meaning" of the word into 3 completely different perspectives: 
      1. **Q** becomes the question (*"I am 'Bank', am I near a 'river'?"*).
      2. **K** becomes the answer key (*"I am 'River', I involve water!"*).
      3. **V** becomes the pure, absolute identity fact of what the word actually is.
    * At this exact nanosecond, `Q`, `K`, and `V` each come out as a massive `128`-decimal array per word (`d_model = 128`). Imagine this as a massive, heavy mental load for a single AI to compute effectively, so we split the work!
  * **The `.view` slice**: The `.view()` instruction mathematically slices apart that 128-decimal array into `n_heads = 4` separate, smaller compartments, placing exactly `d_k = 32` decimals into each compartment! We physically reshape the matrix from `(BatchSize, Words, 128)` into `(BatchSize, Words, 4, 32)`.
  * **The `.transpose(1, 2)` handoff (The Core Secret)**: Why do we transpose (swap) the dimensions? This is the most crucial mathematical requirement in the entire system.
    * **The PyTorch `matmul` Law**: When PyTorch performs Matrix Multiplication (`matmul`), it is strictly hardcoded to ONLY calculate the **last two dimensions** of whatever matrix you hand it. It treats any outer dimensions before that as just a pile of separate, parallel pages.
    * **Before Transpose**: If we left the shape as `(BatchSize, Words, 4 Workers, 32 Decimals)`, the last two dimensions are `Workers` and `Decimals`. If we ran `matmul` on this, PyTorch would try to mathematically multiply "Worker 1" against "Worker 2". This is utter nonsense! The parallel AI Workers aren't supposed to analyze each other; they are supposed to analyze the *Words*!
    * **After Transpose**: By explicitly calling `.transpose(1, 2)`, we physically swap the `Words` dimension (Index 1) with the `4 Workers` dimension (Index 2). The shape magically flips into `(BatchSize, 4 Workers, Words, 32 Decimals)`. 
    * **The Result**: Now, the final two dimensions are perfectly positioned as `Words` and `32 Decimals`. When we run `matmul` on the very next line, PyTorch completely ignores the `BatchSize` and `4 Workers` (treating them like 4 independent clipboards), and flawlessly multiplies the **Words** against the other **Words** simultaneously! This is the exact mechanics of "Multi-Head" Attention: 4 parallel independent workflows calculating word-comparisons at the absolute exact same time without crashing into each other.

```python
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
```
* **Line 47**: This is the most famous equation in AI. It multiplies the Queries (`q`) by the Keys (`k`). If a Query question matches a Key answer, the resulting "score" explodes to a massive number. This means those two words are highly related (like "bank" and "river"). It then divides by the square root of 32 (`sqrt(d_k)`) just to keep the sheer math numbers from spiraling out of control and crashing the GPU.

```python
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
```
* **Lines 49 to 51**: **THE CHEAT BLOCKER.** 
  * Remember, unlike humans reading left-to-right, the AI looks at the *entire* 1024-word flashcard all simultaneously in a fraction of a millisecond. If we are trying to force it to guess Word #5 based on Words 1-4, it will just cheat and physically look at Word #5 on the flashcard! We have to physically blind the workers from looking into the future.
  * The `mask` acts like a giant piece of cardboard covering the future words. The cardboard tells PyTorch exactly which slots contain future words (`mask == 0`). 
  * **Why Negative Infinity (`-inf`)?**: The `masked_fill` function violently overwrites the math score of every single "future" word with literally Negative Infinity! We do this because the very next step is turning these math scores into percentages (`softmax`). If a math score is Negative Infinity, its percentage cleanly instantly becomes **0.0000%**. By permanently erasing the percentage chance of future words down to exactly 0%, it becomes mathematically utterly impossible for the AI workers to look anywhere except the past!

```python
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
```
* **Lines 53 to 54**: **Creating Clean Percentages & Preventing Burnout**
  * **What is `F`?**: On Line 3 of `model.py`, we wrote `import torch.nn.functional as F`. `F` is simply a nickname we gave to PyTorch's mathematical toolbox. By typing `F.`, we are telling Python to reach into that toolbox and pull out a specific math tool.
  * **What is `softmax`?**: Right now, our `scores` from the previous step are wild, massive numbers (like `3.42`, `15.9`, `-inf`). Softmax is a magical mathematical filter. If you pour a bucket of wild numbers into Softmax, it violently squashes them until they perfectly equal exactly `1.00` (which implies 100%).
    * Example: If `"bank"` scored `10.0` with `"river"`, `2.0` with `"dog"`, and `-inf` with a hidden future word, Softmax cleanly converts that to `99.9%` focus on `"river"`, `0.1%` focus on `"dog"`, and exactly `0.00%` focus on the future word! We store these perfect percentages as **Attention Weights** (`attn_weights`).
  * **What is `dim=-1`?**: Imagine the `scores` tensor as a 2D grid where rows are the Words doing the looking (Queries), and columns are the Words being looked at (Keys). The parameter `dim=-1` explicitly tells PyTorch to apply the 100% Softmax mathematically across the **last dimension** (the columns). This means it calculates 100% horizontally across **every single row** independently! It guarantees that each specific word's "Attention" mathematically sums up to perfectly 100% across the rest of the sentence.
  * **Line 54 (`self.dropout`)**: Now we take our perfect percentages and forcefully apply a 10% temporary blindness to them across the board! If we let the AI see perfectly all the time, it becomes lazy and simply memorizes the exam answers without actually learning English grammar. Dropout randomly hides data so the AI is forced to dynamically guess and think on its feet.

```python
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(output)
```
* **Lines 56 to 58**: **Assembling the Final Answer!**
  * **Line 56 (`torch.matmul`)**: Matrix Multiplication. We take our perfect 100% focus percentages (`attn_weights`) and medically multiply them into `v` (the **Values**). 
    * **What is `v`?**: Back on Line 45, we created `v`. While Queries (`q`) asked questions and Keys (`k`) answered them, `v` holds the literal, absolute mathematical definition of what the word actually is! By multiplying the `99.9%` focus percentage by the raw math-meaning of the word `"river"`, we are physically dragging the meaning of "river" and permanently blending it into the meaning of "bank"!
  * **Line 57 (`transpose`, `contiguous`, `view`)**: Once the 4 AI workers have finished reading their 32 smaller slices of meaning, we must stitch those chunks back together into the original 128-slice pizza!
    * **`transpose(1, 2)`**: This perfectly reverses the slice we did on Line 43! We swap the dimensions back, physically prying the clipboards out of the individual 4 workers' hands and laying them flat on the master desk.
    * **`contiguous()`**: When you rapidly slice and flip matrices all over the place in computer memory, PyTorch sometimes scatters the data randomly. `.contiguous()` forcefully sweeps the computer's physical RAM memory so all the numbers are perfectly glued next to each other sequentially, preventing catastrophic memory crashing in the next step.
    * **`view(...)`**: This is the final stitch! We instruct the matrix to forcefully reshape its physical borders to perfectly outline `bsz` (Batch Size: 4 flashcards), `seq_len` (1024 words per flashcard), and `self.d_model` (the fully assembled 128 decimals of meaning)! The full pizza is officially fully reunited.
  * **Line 58 (`self.out_proj`)**: Before we return the final constructed answer, we safely pass it through `out_proj`, which is one final neural network filter (spawned on Line 36). It behaves like a safety sander, softly mathematically polishing out any rough edges in the stitched-together data before violently sending it up to the next floor of the Skyscraper!

---
### 4. The Transformer Block (The Floor of the Building)
This block combines the Attention mechanism from above with the "private scratchpad" memory we discussed earlier (`d_ff`).

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = Attention(d_model, n_heads, dropout)
```
* **Lines 60 to 63**: We build a Floor. We install the Attention mechanism onto the floor.

```python
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
```
* **Lines 64 to 70**: **The Feed-Forward Network (The AI's Private Scratchpad)**
  * **What is `nn.Sequential`?**: This is PyTorch's elegant way of creating an assembly line. Instead of writing out the flow of data manually step-by-step, we package 5 distinct neural network operations into a single tube. When a word enters `self.ffn`, it will automatically ride this 5-step assembly line sequentially from top to bottom.
  * **Step 1 (`nn.Linear(d_model, d_ff)`) - The Expansion**: First, the word enters with `128` decimals of meaning. This network powerfully expands it, projecting the 128 decimals into a massive `512`-decimal private workspace (`d_ff = 512`). Why? Because the `Attention` mechanism we just modeled was only about *collecting context from other words*. The AI actually mathematically needs a massively wide private scratchpad to "think" deeply about what that collected context actually *means* for the current word. 
  * **Step 2 (`nn.GELU()`) - The Brain Spark**: Matrices themselves are just flat, boring algebra (`Y = aX + b`). To make them actually learn complex curves (like the intricate rules of human languages), we must introduce non-linearity. `GELU` is the mathematical spark that mimics physical brain neurons firing. It looks at the 512 numbers on the scratchpad; if a number is strongly negative, it mathematically deletes it (forcing the neuron to "turn off"). If a number is positive, it safely lets it pass through unhindered (forcing the neuron to excitedly "fire").
  * **Step 3 (`nn.Dropout`) - The Forgetfulness Trick**: Once again, we apply 10% temporary blindness to the firing neurons. If we let the AI safely rely on the exact same neurons every single time, it gets lazy and memorizes the Wikipedia dataset. By randomly shutting off 10% of the active neurons on the scratchpad, the AI is mercilessly forced to distribute its knowledge adaptively across all 512 pathways!
  * **Step 4 (`nn.Linear(d_ff, d_model)`) - The Compression**: Now that the heavy thinking is fully complete, the 512-decimal scratchpad is far too massively bloated to fit through the exit doors of the floor. This network rigorously shrinks the 512 deeply processed decimals back down into a perfectly neat, highly concentrated `128`-decimal summary.
  * **Step 5 (`nn.Dropout`) - Final Blindness**: Another quick 10% blindness is applied to the newly compressed 128-decimal array just before it officially exits the scratchpad module to ensure it doesn't over-confidently rely on a single prominent data point.

```python
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
```
* **Lines 71 & 72**: **The Bouncers (`LayerNorm`)**. You might intimately wonder why `LayerNorm` is kept physically outside the `self.ffn` sequential assembly line we just tightly wrapped up. The underlying reason is structural: `LayerNorm` isn't conceptually a part of the Attention room, nor is it a part of the Scratchpad room. It acts strictly as a Bouncer heavily guarding the doors of *both* rooms. If the complex matrix math ever gets thousands of times too loud from previous calculations, the Bouncer violently compresses the mathematical volume back down to a safe baseline *before* safely letting the data walk inside the room.

```python
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.ffn(self.ln2(x))
        return x
```
* **Lines 74 to 77**: **The Master Blueprint & Residual Connections**. This function physically dictates how data legally moves through the Transformer Floor, and it reveals the single most vitally important mathematical backbone in the entire building: the **Residual Connection** (`x = x + ...`).
  * **Step A**: The raw 128-decimal flashcard (`x`) attempts to enter the Attention room. It is strictly intercepted by the Bouncer (`self.ln1(x)`). The safe, volume-corrected data securely enters the Attention room (`self.attn(...)`) where it deeply gathers contextual focus. Finally, it exits the room and mathematically adds itself completely back to the *raw, strictly unaltered original flashcard* (`x + ...`)!
  * **Step B**: The newly context-heavy word now attempts to enter the Scratchpad. Once again, it is strictly volume-checked by the second Bouncer (`self.ln2(x)`), it undergoes the massive, rigorous 5-step "Expand-Compute-Shrink" process inside (`self.ffn(...)`), and when it conclusively comes out, it again mathematically adds itself back strictly to the original word!
  * **Why Bypass?**: Adding the raw data mathematically back to itself creates a literal physical "Bypass Highway" for the data. By completely skipping the heavy processing rooms, the AI is mathematically hard-sealed and physically guaranteed to never accidentally "forget" or alter the original core word while simultaneously doing complex deep-thinking calculations!

---
### 5. Assembling the Final Model
This puts everything together into one giant wrapper.

```python
class TransformerLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = PositionalEncoding(config.d_model, config.max_seq_len)
        self.drop = nn.Dropout(config.dropout)
```
* **Lines 79 to 85**: **Building the Master Skyscraper (`TransformerLM`)**
  * **Lines 79 & 80**: We officially define the master `TransformerLM` class. Just like the smaller blocks, it requires PyTorch's `nn.Module` setup. We securely pass in the master `config` object so the Skyscraper knows exactly how tall and wide it theoretically needs to be built.
  * **Line 81 (`super().__init__()`)**: The standard Python object-oriented rule. We rigorously run PyTorch's base setup sequence first to officially register this custom skyscraper with the GPU's deep memory manager.
  * **Line 82 (`self.config = config`)**: We physically glue the master configuration blueprint directly onto the Skyscraper itself, so that any underlying function can reference the master blueprint variables at any given time.
  * **Line 83 (`nn.Embedding`)**: This is the absolute core Translation Dictionary! Remember `data.py`? It rigidly converted the English word `"Apple"` into the flat integer `4201`. However, Neural Networks cannot multiply single integers; they strictly require massive 128-decimal matrices! `nn.Embedding` automatically generates a giant lookup table with 50,257 rows (Vocab Size) and 128 columns (d_model). When you physically pass the integer `4201` into it, it instantly looks up row 4201 and returns the 128-decimal mathematical "meaning" vector that specifically corresponds to "Apple".
  * **Line 84 (`PositionalEncoding`)**: Because the dictionary (`nn.Embedding`) only provides the theoretical *meaning* of the word, it completely lacks any context of *where* the word physically sits in the sentence. Here, we officially spawn the `PositionalEncoding` tool we meticulously built way back on Line 6. We will soon use this exact tool to stamp mathematically unique sine-wave GPS coordinates directly onto those 128 decimals.
  * **Line 85 (`nn.Dropout`)**: A 10% blindness tool specifically reserved for the main front door of the Skyscraper. When words physically first enter the building and get their GPS coordinates heavily stamped, we instantly blind 10% of their starting decimals so the network is violently forced to rely on dynamic contextual grammar right from the very first ground level.

```python
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
```
* **Lines 87 to 90**: It builds the skyscraper! Because `n_layers = 2`, it loops twice, physically stacking two `TransformerBlock` floors on top of each other.

```python
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
```
* **Lines 91 & 92**: **The Final Polish and the Output Soundboard**
  * **Line 91 (`self.ln_f`)**: After the data has cleanly ridden the elevator all the way up through the multiple stacked Transformer floors, it reaches the roof. Before we attempt to translate the complex math back into English words, we run the 128-decimals through one final "Bouncer" (`LayerNorm`). This mathematically guarantees the final numbers are perfectly smooth and level before the delicate translation process.
  * **Line 92 (`self.head`)**: This is the massive Output Soundboard we first referenced way back in `data.py`! The AI has finally finished all its deep thinking, and it now firmly holds a highly processed 128-decimal concept of what it assumes the very next word should be. However, humans can't read 128 decimals. `self.head` is a massive `nn.Linear` network that violently un-compresses those `128` concentrated concept-decimals cleanly back out into **`50,257` raw probability scores** (`vocab_size`)! Every single individual score maps directly to a specific English word button in the OpenAI dictionary! 

```python
        # Weight tying
        self.token_emb.weight = self.head.weight
```
* **Lines 94 & 95**: **The "Weight Tying" Master Stroke**
  * **The Problem**: Look closely at `token_emb` (Line 83) and `head` (Line 92). `token_emb` physically translates `50,257 -> 128`. The `head` structurally translates the exact opposite: `128 -> 50,257`. Both of these neural networks contain exactly 6,432,896 massive weight connecting decimals (`128 * 50257`), taking up astronomical chunks of limited GPU memory.
  * **The Concept**: Translating "Apple" into math should theoretically require the exact same mathematical logic as translating math back into "Apple". By manually forcefully instructing the output soundboard (`self.head.weight`) to physically share the exact same underlying block of computer RAM as the input dictionary (`self.token_emb.weight`), we mathematically construct a two-way street. 
  * **The Result**: 
    1. **Memory**: We instantly permanently delete over 6.4 million redundant mathematical parameters from the neural network, freeing up massive amounts of physical VRAM!
    2. **Learning Speed**: It drastically heavily accelerates the AI's learning curve. If the AI struggles and finally learns a deep grammatical rule about the word "Dog" while *reading* the textbook, the output soundboard instantaneously mathematically learns how to optimally *write* the word "Dog" on the test because their conceptual brains are physically glued together!

```python
    def _generate_causal_mask(self, seq_len, device):
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.view(1, 1, seq_len, seq_len)
```
* **Lines 97 to 99**: **Building the "Cheat Blocker" Triangle**
  * **Line 98 (`torch.ones`)**: We mathematically create a giant square grid of pure `1`s. We deliberately pass `seq_len` twice (`seq_len, seq_len`) because when analyzing a sentence, every single Word (Rows) must identically be cross-checked against every single other Word (Columns)! If the flashcard has 1,024 words, we physically need a massive 1,024 x 1,024 grid to aggressively cross-check every possible contextual combination.
  * **Line 98 (`torch.tril`)**: The magic command. "Tril" stands for **Tri**angle **L**ower. It aggressively cuts the square matrix precisely diagonally in half like a sandwich. Everything in the *Lower* Triangle securely stays as a `1` (which signifies the "past" words we are legally allowed to look at). Everything in the *Upper* Triangle is instantly violently converted to `0`s (which signifies "future" words we are temporarily legally forbidden from seeing).
  * **Line 99 (`.view`)**: We officially force the `view()` method to elegantly shape the flat 2D triangle mask into an explicit 4-dimensional object: `(1, 1, seq_len, seq_len)`. Why these exact positions?
    * Because earlier in the Attention room (Line 47), our transposed parallel math perfectly became an exact 4-dimensional array: `(Batch_Size, Num_Workers, Words, Words)`.
    * If we naively try to physically violently overlay a flat 2D mask matrix on top of a complex 4D brain structure, PyTorch mathematically crashes instantly! The shapes must technically map to each other perfectly.
    * **Broadcasting (`1, 1`)**: By strategically sliding a `1` into the very first two dimension slots (`Batch_Size=1, Num_Workers=1`), we elegantly trigger a secret PyTorch acceleration trick called **Broadcasting**. PyTorch instantly realizes it doesn't need to manually, painfully construct independent masks from scratch for every single worker; it just effortlessly mathematically copy-pastes this identically shaped masking square natively across *all* parallel worker domains simultaneously!

```python
```python
    def forward(self, x):
        bsz, seq_len = x.size()
        mask = None
        # Linear attention handles causality internally, doesn't need external mask
        if self.config.attention_type != "linear":
            mask = self._generate_causal_mask(seq_len, x.device)
```
* **Lines 104 to 109 (The Front Door)**: **[Task 2 Change]** When words enter the building, we usually build a "Cheat Blocker" triangle (`mask`). However, **Linear Attention** (the variant we built in Task 2) inherently knows not to look into the future based on its internal math! Therefore, we added these lines so the model can intelligently skip the expensive masking step if we are in Linear mode, making it much faster.

```python
        x = self.token_emb(x)
        x = self.pos_emb(x)
        x = self.drop(x)
```
* **Lines 105 to 107**: **The Lobby Preparation Step**
  * The raw flat integers (`x`) step up to the Lobby desk (`token_emb`) and are beautifully translated into vast, massive 128-decimal concept arrays.
  * They securely step over to the printing station (`pos_emb`) to get their precise Sine-Wave GPS structural coordinates aggressively stamped directly into their decimals so they know their exact mathematical chronological order.
  * They pass through the Front Door Scanner (`drop`) which temporarily blinds exactly 10% of their decimals so the network is violently forced to rely on dynamic grammatical context instead of lazily memorizing the Wikipedia paragraphs straight away.

```python
        for block in self.blocks:
            x = block(x, mask=mask)
```
* **Lines 109 to 110**: **Riding the Elevator Up!**
  * **The Massive "For Loop"**: The data formally enters the main Skyscraper elevator. `n_layers` dictated exactly how many specific floors we physically built in the Skyscraper (in our config, we built `2`). 
  * The heavily prepared data (`x`) strictly enters Floor 1 (`block`). It gets heavily processed, mathematically twisted, and meticulously filtered. 
  * When Floor 1 flawlessly finishes, its exact mathematical output spectacularly physically becomes the brand-new `x` input for Floor 2! We simply aggressively ride the data looping sequentially all the way up through the entire building. Notice we consciously securely pass the `mask=mask` Cheat Blocker up into every single floor layout!

```python
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
```
* **Lines 112 to 114**: **The Final Output (The Skyscraper's Roof)**
  * **Line 112 (`self.ln_f`)**: The deeply complex context data securely exits the final Floor 2 elevator and reaches the roof. Before translating it, it encounters the very last mathematical Bouncer (`LayerNorm`), which carefully physically smooths out the math volumes one final time.
  * **Line 113 (`self.head`)**: We violently push the level 128-decimals directly into the Output Soundboard (`head`). The soundboard aggressively mathematically un-compresses those `128` deep structural context decimals precisely back into exactly **50,257** individual probability buttons!
  * **Line 114 (`return logits`)**: These 50,257 mathematically-calculated probability scores are commonly referred to as **Logits**. The Skyscraper throws these raw Logits solidly back out the front door, directly into the hands of the Teacher (`train.py`), who will rigorously mathematically grade the AI's guesses and brutally adjust its internal brain Weights if it failed the exam!

---

### 🧠 TASK 3 CHANGES: The Positional Factory

In Task 3, we upgraded the Skyscraper to handle much longer books by switching to **Relative** positions.

```python
    def forward(self, x, mask=None, pos_params=None):
```
* **[TASK 3 CHANGE] (Line 37 & 77)**: We updated the `forward` pass of **StandardAttention** and the **TransformerBlock**. They now accept a secret signal called `pos_params`. This is a mathematical "Envelope" that contains either the **Clock Spins** (RoPE) or the **Distance Penalties** (ALiBi).

```python
    def _get_rope_embeddings(self, seq_len, device):
```
* **[TASK 3 CHANGE] (Lines 100 to 110)**: This is a new "Caching Station." Calculating the **RoPE** trigonometry (Sin/Cos) is expensive. Instead of doing it every single step, the model now calculates it once for the maximum length and "caches" (saves) it in its memory. If the next sentence is shorter, it just looks up the answer in its cache!

```python
        # [Task 3] Handle Positional Logic
        if self.config.pos_type == "rope":
            cos, sin = self._get_rope_embeddings(seq_len, x.device)
            pos_params["cos"], pos_params["sin"] = cos, sin
        elif self.config.pos_type == "alibi":
            pos_params["alibi_bias"] = build_alibi_bias(self.config.n_heads, seq_len, x.device)
```
* **[TASK 3 CHANGE] (Lines 118 to 123)**: This is the "GPS Dispatcher." 
    * If you chose **RoPE**, it fetches the Sin/Cos clocks.
    * If you chose **ALiBi**, it generates the "fading signal" penalty grid.
    * It then neatly packs these into the `pos_params` envelope to send up the elevator.

```python
        # Only use absolute embeddings if not using modern variants
        if self.config.pos_type == "absolute":
            x = self.pos_emb(x)
```
* **[TASK 3 CHANGE] (Lines 128 to 130)**: We added a "Security Gate." **RoPE** and **ALiBi** work *inside* the attention layers, so they don't need the Absolute page numbers from Task 1. This `if` statement ensures we only use the old page numbers if the user specifically asked for them, preventing the math from getting double-stamped and corrupted!

---

### 🏛️ FINAL TASK 3 DEEPER DIVE: Lines 111 to 157

This is the most critical part of your new architecture—the **Logistics Center**.

#### 1. The Cheat Blocker (Causal Mask)
```python
111: def _generate_causal_mask(self, seq_len, device):
112:     mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
113:     return mask.view(1, 1, seq_len, seq_len)
```
* **Line 112 - The Triangle**: `torch.ones` creates a square grid. `torch.tril` (Triangle Lower) cuts it precisely in half diagonally.
  * **Analogy**: It’s a "Time Shield." It makes all future words **zero** so the AI can only "see" into the past.
* **Line 113**: We add a `1, 1` dimension. This is the **Broadcasting Adapter** that allows one single mask to be copy-pasted across all 4 Heads and the entire Batch instantly.

#### 2. The Frozen GPS (RoPE Cache)
```python
115: def _get_rope_embeddings(self, seq_len, device):
117:     if not hasattr(self, 'rope_cache') or self.rope_cache['cos'].size(0) < seq_len:
```
* **Line 117 - The Freezer Check**: The AI checks: *"Do I already have frozen clocks in my freezer (`rope_cache`)? And are they big enough for this long book?"* If yes, it skips the expensive math!
* **Line 119 - The Pulse Range**: This calculates how "fast" the clocks spin. Some spin fast for nearby words; others spin slow for big context.
* **Line 121 - The Map**: We combine **Positions** with **Frequencies** to create the final "Clock Angle Map."
* **Line 122**: We pre-calculate all **Sines** and **Cosines** and freeze them in a cache so the model never has to do this work again during training.

#### 3. The Master Forward Loop (The Elevator)
```python
131:     pos_params = {}
```
* **Line 131 - The Envelope**: We prepare a secret dictionary called `pos_params`.

```python
134:     if self.config.attention_type != "linear":
135:         mask = self._generate_causal_mask(seq_len, x.device)
```
* **Line 134 - The Intelligence Check**: Your model knows that **Linear Attention** (Task 2) has its own internal way of hiding the future, so it smartly skips building the `mask` to save memory! 

```python
138:     if self.config.pos_type == "rope":
139:         cos, sin = self._get_rope_embeddings(seq_len, x.device)
140:         pos_params["cos"], pos_params["sin"] = cos, sin
141:     elif self.config.pos_type == "alibi":
142:         pos_params["alibi_bias"] = build_alibi_bias(...)
```
* **Lines 138-142 - Filling the Envelope**: This is the **Dispatcher**. Based on your config, it grabs the **Clocks** (RoPE) or the **Penalty Grid** (ALiBi) and stuffs them into the `pos_params` envelope.

```python
147:     if self.config.pos_type == "absolute":
148:         x = self.pos_emb(x)
```
* **Line 147 - The Legacy Gate**: We only use the old page-numbers if the user specifically asked for them.

```python
152:     for block in self.blocks:
153:         x = block(x, mask=mask, pos_params=pos_params)
```
* **Line 153 - Passing the Signal**: The data rides the elevator up every floor. Each floor receives the **Cheat-Blocker** (`mask`) and the **GPS Envelope** (`pos_params`).

```python
156:     logits = self.head(x)
```
* **Line 156**: The final transformation. We turn our 128-decimal concept vectors back into **50,257** individual word probability scores.
