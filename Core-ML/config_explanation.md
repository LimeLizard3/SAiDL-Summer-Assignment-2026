# Beginner's Guide to `config.py`

Before building a house, the builders need to know exactly how many windows, doors, and rooms it will have. Instead of scattering these measurements all over the construction site, they put them on a single blueprint document so everyone knows exactly what the plan is.

That is what `config.py` does for our Artificial Intelligence model. It defines the size, shape, and learning style of our AI in one central place.

---

### Setting up the Blueprint

```python
from dataclasses import dataclass
```
* **What it means:** Python comes with built-in helper tools, but leaves them switched off by default to save memory. This line tells Python: *"Fetch the `dataclass` tool from your toolbox so I can use it."*

```python
@dataclass
class TransformerConfig:
```
* **What it means:** A `class` creates a group. We are creating a group named `TransformerConfig`. The `@dataclass` line tells Python to strictly organize this group into a clean, easy-to-read checklist.

---

### Defining the Data Rules

```python
    vocab_size: int = 50257
```
* **What it means:** This tells the AI how many unique "words" exist in its dictionary. Because computers only understand numbers, we have to translate all human text into numbers. This means our AI understands exactly `50,257` unique numbers (or "tokens").

```python
    max_seq_len: int = 1024
```
* **What it means:** This sets the "working memory" limit of our AI. It tells the AI it is only allowed to look at `1024` words at a time when trying to predict what word comes next.

---

### Defining the Size of the AI's "Brain"

```python
    d_model: int = 128
```
* **What it means:** How big is the mathematical "concept" for a single word? When the AI reads the word "Dog", it converts it into a list of exactly `128` decimal numbers that represent deeper meaning.

```python
    n_heads: int = 4
```
* **What it means:** How many different "perspectives" the AI uses to stare at a sentence simultaneously. Here, it has `4` different attention mechanisms sweeping over the text at the same time to look for complex patterns.

```python
    n_layers: int = 2
```
* **What it means:** We are stacking `2` of these mathematical brains on top of each other. The first layer might notice simple grammar, and then passes its notes up to the second layer, which notices deeper themes.

```python
    d_ff: int = 512
```
* **What it means:** After the AI figures out how words relate to each other, it passes that information into a traditional "Feed-Forward" neural network to process the final logic. This sets the size of that processing area.

```python
    dropout: float = 0.1
```
* **What it means:** This literally tells the AI to randomly turn off `10%` (`0.1`) of its brain cells while it is studying. This mathematically forces it to adapt and generalize instead of just cheating and memorizing the answers.

---

### 🧠 Task 2 Change: Architecture Variants

```python
    attention_type: str = "mqa" 
```
* **What it means (Line 14):** **[Task 2 Change]** This is the master "Switch." It tells the model which mathematical attention engine to use. You can choose between `"standard"`, `"mqa"`, `"linear"`, or `"sliding_window"`. Each one calculates context differently to save speed or memory!

```python
    window_size: int = 50
```
* **What it means (Line 15):** **[Task 2 Change]** This is only used if you set the Switch to `"sliding_window"`. It tells the AI exactly how many words back it is allowed to look. If it's `50`, the AI can only "remember" the immediate 50 words behind the one it's currently guessing.

---

### Defining How the AI Learns (The Training)

```python
    batch_size: int = 4
```
* **What it means:** Instead of making the AI read one single sentence and update its brain, we make it read a "batch" of `4` completely different text chunks at the exact same time before it is allowed to correct its mistakes.

```python
    learning_rate: float = 3e-4
```
* **What it means:** When the AI gets an answer wrong during training, it tweaks its internal math to try and get it right next time. This sets exactly how "big" of an adjustment it is allowed to make (`3e-4` is a tiny fraction: `0.0003`).

```python
    epochs: int = 1
```
* **What it means:** An `epoch` is one full read-through of the entire textbook (dataset). `1` means the AI will read all the data from start to finish exactly one time.

```python
    weight_decay: float = 0.01
```
* **What it means:** A mathematical punishment tool. Over time, AI models like to make their internal math numbers massive to feel falsely confident about their answers. This tells the system to constantly shrink (`decay`) the numbers by a tiny `1%` to keep the math healthy and stable.

---

### Additional Explanations (Q&A)

#### What exactly is the purpose of `d_ff = 512`?
In a Transformer, every single "Layer" has two critical halves that work as a team:
1. **The Attention Mechanism:** This is the part that scans the sentence, looks at how all the words relate to one another, and gathers context. 
2. **The Feed-Forward Network:** This is what `d_ff` controls. 

If Attention is the part of the brain that **reads and gathers information**, the Feed-Forward Network is the part that **stops, thinks, and stores facts**. 

Instead of just looking at relations between words, the `d_ff` (Feed-Forward) section acts like a huge private scratchpad where the AI individually processes what each word means based on the new context it just gathered. The number `512` dictates how massive that private scratchpad is. The bigger this number, the more raw facts, grammar rules, and logic the AI can memorize and process per word!

#### So we have `4` heads and `2` layers. Does that mean `8` attention mechanisms?
**Yes! Exactly.**

Imagine an office building with exactly 2 floors (`n_layers = 2`).
* **Floor 1 (Layer 1):** Has 4 workers (`n_heads = 4`). They are sitting at their desks, looking intently at the actual text you typed in. One worker might be looking exclusively for nouns, another for verbs, and another tracking emotional tone.
* **Floor 2 (Layer 2):** Also has 4 workers (`n_heads = 4`). However, they do *not* look at the raw text. Instead, Phase 1 workers hand their "notes" up to the second floor. The Phase 2 workers use their 4 attention mechanisms to look at the *notes* from the first floor, allowing them to draw much deeper, more complex conclusions (like spotting sarcasm or solving a math puzzle).

So your model has **8 independent attention workers** operating in total: 4 on the first floor analyzing the raw words, and 4 on the top floor acting as the managers summarizing everything!

---

### 🧠 TASK 3 CHANGES: Positional Variants

```python
    pos_type: str = "absolute"
```
* **What it means (Line 18):** **[TASK 3 CHANGE]** This is the modern "GPS Switch." It tells the model how to understand where words are in a sentence relative to each other. 
    * **`"absolute"`**: Every word gets a fixed page number (1-512). It works, but it breaks if the book gets too long.
    * **`"rope"`**: **Rotary** positional embeddings. It rotates word vectors in a circle to show their position. It's used by the world's best models like **Llama 3**.
    * **`"alibi"`**: Adds a mathematical penalty that makes distant words grow weaker in the AI's mind, allowing it to read books of any length without getting confused.
