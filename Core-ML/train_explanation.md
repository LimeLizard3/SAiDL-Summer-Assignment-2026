# The Teacher: `train.py` (Line-by-Line Breakdown)

If `model.py` is the isolated Brain of the AI, then `train.py` is the rigid **Master Teacher** and the **Classroom**. This file builds the brain, feeds it the flashcards, brutally grades its tests, and forcefully tweaks its neurons so it learns.

Let's break down exactly how the Teacher mathematically forces the AI to learn English grammar.

---
### 1. The Silently Brilliant Auto-Installer & Imports
Before the AI can even begin, we have to precisely prepare the classroom.

```python
import sys
import subprocess

def auto_install():
    packages = ["torch", "datasets", "tiktoken"]
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing missing package: {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

auto_install()
```
* **Lines 1 to 13 (`auto_install`)**: Before the script begins, the Teacher quickly conceptually checks the student's backpack. It rigorously tries to globally import the 3 massive libraries we absolutely need (`torch` for math, `datasets` for wikipedia, `tiktoken` for the dictionary). 
  * If the mandatory import dynamically fails (`ImportError`), it explicitly doesn't violently crash the script! Instead, it secretly strictly opens the computer's terminal completely in the background mathematically using `subprocess.check_call`. 
  * Under the hood, `sys.executable` perfectly tracks down exactly which invisible version of Python is officially running this script, and it strictly forcefully commands it to natively run `-m pip install` to safely automatically download your missing packages mechanically! This is a genius quality-of-life trick so your exact code cleanly runs effortlessly on any computer without screaming errors.

```python
import torch
import torch.nn.functional as F
import time
import math
from data import get_dataloaders
from model import TransformerLM
from config import TransformerConfig
```
* **Lines 15 to 21 (The Core Imports)**: Now that the packages are successfully installed, we seamlessly physically import the actual functionality. 
  * `torch` and `F` rigidly bring in the massive matrix math engine.
  * `time` strictly acts as the Teacher's stopwatch (so we can precisely mathematically calculate how aggressively fast the AI is natively answering flashcards).
  * `math` natively gives us advanced geometric formulas (like exponentiation `math.exp()`) to accurately mathematically calculate the final exam grade.
  * Finally, we logically structurally import the three exact puzzle pieces we successfully built in the previous files: The Blueprints (`TransformerConfig`), the Student Desks (`get_dataloaders`), and the Skyscraper Brain (`TransformerLM`).

---
### 2. The Final Exam (The Grading Rubric)
Even though the actual Final Exam happens at the very end of the script, the Teacher strictly explicitly defines the "Grading Rubric" upfront first chronologically.

```python
def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for x, y in val_loader:
             x, y = x.to(device), y.to(device)
             logits = model(x)
             loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='sum')
             total_loss += loss.item()
             total_tokens += y.numel()
```
* **Lines 23 to 34 (The Honest Exam)**: This visually natively looks exactly like the training loop, but with two absolutely critical, life-or-death structural differences heavily at the beginning:
  * **Line 24 (`model.eval()`)**: We effectively formally brutally turn OFF the 10% Dropout blindness. The AI is aggressively strictly taking a life-or-death final exam; it mathematically definitively needs 100% full raw access to all its neurons safely without being artificially strategically blinded!
  * **Line 27 (`torch.no_grad()`)**: We forcefully physically deeply strictly freeze PyTorch's "Autograd Engine!" This formally structurally legally bans the Teacher from pulling out the Red Pen and dynamically editing the AI's neuronal weights rigidly during a test. By explicitly forcefully turning off the massively heavy backward-tracking math, the final exam natively dynamically runs 10x faster and requires rigidly 50% less physical Graphics Card memory!
  * **Lines 28 to 31**: The Teacher secretly heavily hands the AI the `val_loader` desk. These are explicit, hidden Wikipedia paragraphs the specific AI has **never** visually mathematically seen before! We rigorously use the exact identical `F.cross_entropy` grading formula to specifically objectively check the AI's percentage guesses. However, precisely because `no_grad()` is turned legally on, the Teacher silently mathematically writes the `loss` penalty on a piece of paper but explicitly refuses to permanently twist the brain knobs.
  * **Line 33 (`total_tokens += y.numel()`)**: We meticulously rigidly effectively objectively count exactly how many words (`y.numel()`) the AI was specifically structurally tested on so we can perfectly mathematically calculate a fair final average grade later.

```python
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    return avg_loss, perplexity, inference_throughput
```
* **Lines 23 to 43 (The Honest Exam & Speedometer)**: **[Task 2 Change]** This is where the Teacher grades the AI. In Task 2, we added a "Stopwatch" (`start_time` and `end_time`) to the exam.
  * **Line 39 (`inference_throughput`)**: We calculate your **Inference Speed**. We count all words tested and divide by the seconds elapsed. This gives us "Tokens Per Second"—a critical metric for measuring how fast our new attention variants actually are!

---
### 3. The Master Setup Phase
```python
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = TransformerConfig()
    print(f"Running Experiment: Attention={config.attention_type} | SeqLen={config.max_seq_len}")
```
* **Line 50 (The Banner)**: **[Task 2 Change]** We added this print statement so you can immediately see which attention engine and sequence length you are currently testing right when you hit run!

```python
    train_loader, val_loader, vocab_size = get_dataloaders(
        seq_len=config.max_seq_len, 
        batch_size=config.batch_size
    )
    config.vocab_size = vocab_size
```
* **Lines 43 to 49**: **Loading the Desks**. 
  * **Line 43**: We physically grab the master skyscraper blueprint.
  * **Line 45**: We summon `get_dataloaders` from `data.py`. Remember what this did? It sliced Wikipedia into 1,024-word flashcards and bundled them into groups of `4` (the Batch Size). We receive the `train_loader` (the study deck) and the `val_loader` (the hidden test deck).
  * **Line 49**: We dynamically overwrite the blueprint's dictionary size to perfectly match whatever dictionary OpenAI generated (`50,257`).

```python
    model = TransformerLM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
```
* **Lines 51 to 58**: **Spawning the AI**
  * **Line 51 (`.to(device)`)**: We strictly follow the `config` blueprint and permanently physically spawn the massive `TransformerLM` Skyscraper in computer memory. The `.to(device)` command is critical! It aggressively physically transports the entire 6-million-parameter Skyscraper directly out of standard computer RAM and violently drops it into the hyper-fast VRAM of your Graphics Card!
  * **Line 52 (`optimizer = ... AdamW`)**: The **Optimizer** is literally the Teacher Algorithm. If the AI makes a bad guess later, `AdamW` is the specific algorithm that reaches into the Brain, aggressively grabs the guilty neuronal weights, and violently twists them so the AI theoretically doesn't make that mistake again. 
  * **Line 53 (`model.parameters()`)**: We hand the Teacher a master list of all 6.4 million twistable mathematical knobs in the Brain.
  * **Line 54 (`lr=learning_rate`)**: The "Learning Rate" dictates strictly how violently the Teacher twists the knobs. If it twists too fast, the brain shatters. If it twists too slow, the AI takes 100 years to learn the word "Apple."
  * **Line 58 (`sum(...)`)**: A genius line of code that asks PyTorch to physically count every single neuronal knob (`numel()`) in the brain. It mathematically divides by 1 million (`1e6`) and prints: `"Model parameters: 6.4M"`.

---
### 4. The Study Hall (The Training Loop)
```python
    epochs = config.epochs
    for epoch in range(epochs):
        model.train()
```
* **Lines 60 to 63**: An `epoch` means reading the *entire* Wikipedia textbook comprehensively exactly once. We tell it to loop exactly 3 times (`epochs = 3`).
* **Line 63 (`model.train()`)**: This is incredibly important! This clicks a master safety switch that fully visually activates the **10% Dropout** blindness tools we built earlier! We explicitly want the AI to be blind and struggle while studying so it doesn't just legally memorize the book. 

```python
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
```
* **Line 68 (`enumerate` loop)**: The `train_loader` structurally passes out groups of flashcards (based on `batch_size`). `enumerate` adds a counter (`i`) acting as our **Step** number for progress tracking. `x` is the Inputs ("Front" of the flashcard) and `y` is the Targets ("Back" of the flashcard containing the correct expected answers).
* **Line 69 (`x.to(device)`)**: The CPU standard memory physically stores the `train_loader` flashcards to prevent the Graphics Card from instantly overflowing with an Out-of-Memory error. Right before the AI needs them, we `.to(device)` to safely teleport *only the current specific small batch of flashcards* completely out of standard RAM directly into the hyper-fast VRAM of the Graphics Card so the math can happen instantly.

```python
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
```
* **Lines 71 to 73**: **The Guess and the Red Pen**
  * **Line 71 (`zero_grad`)**: Wiping the chalkboard. Before the AI makes its guess, the Teacher completely erases any lingering penalty points from the previous flashcard so this new guess is judged independently.
  * **Line 72 (`model(x)`)**: The AI makes its guess! It shoves the flashcards `x` up the Skyscraper elevator, and receives back **`logits`**. Logits are the 50,257 raw, unpolished mathematical output scores (like pressure readings on the soundboard) before they are fully converted into clean 0-100% probabilities.
  * **Line 73 (`cross_entropy` and `.view(-1)`)**: The Teacher's brutally grading red pen. `cross_entropy` mathematically calculates the exact "Loss" (Penalty Grade) by comparing the raw `logits` against the correct answer (`y`). 
    * **Why `.view(-1)`? (The Flattener):** PyTorch's grader is incredibly strict; it refuses to grade massive 3D blocks of data and demands a simple 2D spreadsheet. Because we fed the AI 4 flashcards (`batch_size = 4`) that are each 1,024 words long (`seq_len = 1024`), the AI had to make a total of `4 x 1024 = 4,096` individual guesses. 
    * `y.view(-1)` violently flattens the 2D grid of target answers into one single, massively long list of 4,096 target words.
    * `logits.view(-1, logits.size(-1))` precisely flattens the massive 3D brick of mathematical guesses into a 2D spreadsheet: exactly 4,096 rows tall, and 50,257 columns wide (preserving the dictionary probability buttons). 
    * Now, `cross_entropy` is happy! It checks Row 1 against Target Word 1, calculates the exact penalty, and flawlessly does it 4,095 more times in a row!

```python
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
```
* **Lines 74 to 76**: **The Learning Mechanism**
  * **Line 74 (`loss.backward()`)**: The absolute most important line of code in deep learning. This triggers PyTorch's "Autograd Engine". It fires completely backward down through the entire Skyscraper elevator, intricately tracking precisely *which* of the 6.4 million neuronal knobs mathematically caused the bad guess!
  * **Line 75 (`clip_grad_norm_`)**: A surgical safety net preventing a mental breakdown. If the penalty assigned by Autograd is catastrophically massive, this physically clips the severity down to `1.0` so the Teacher mathematically doesn't violently break the brain by twisting the knobs too hard.
  * **Line 76 (`optimizer.step()`)**: **The exact moment of learning!** `AdamW` structurally physically twists the guilty neuronal knobs in the exact reverse direction of the penalty, permanently mutating the brain! The AI has officially organically learned from its mistake!

```python
86:             tokens_processed += y.numel()
```
* **Line 86 (`tokens_processed`)**: **[Task 2 Change]** This is our "Token Odometer." `y.numel()` counts exactly how many words (tokens) were just processed in this specific heartbeat of training (for you, it's 4,096). We add this to our running total so we can later calculate how fast the AI is "reading."

```python
88:             if i > 0 and i % 50 == 0:
```
* **Line 88 (`i % 50`)**: This is the Teacher "Checking In." Instead of printing a million lines of text, it only stops once every 50 batches to tell you how it's going. 

```python
89:                 elapsed = time.time() - start_time
90:                 throughput = tokens_processed / elapsed
```
* **Lines 89 & 90 (The Speedometer)**: **[Task 2 Change]**
    *   **Line 89 (`elapsed`)**: We subtract the current "Now" time from our earlier "Start" time. This tells us exactly how many seconds have passed since the last Check-In. 
    *   **Line 90 (`throughput`)**: This is how we find the **Training Speed**. We take the total words read (`tokens_processed`) and divide by the seconds (`elapsed`). It outputs "Words Per Second." 

```python
91:                 avg_train_loss = total_loss / 50
92:                 train_ppl = math.exp(avg_train_loss) if avg_train_loss < 20 else float('inf')
```
* **Lines 91 & 92 (Training Grade)**: We calculate the average penalty (Loss) over the last 50 steps and turn it into **Perplexity (PPL)**. Perplexity is the AI's "Confusion Score." 1.0 is a perfect score (zero confusion); a huge number means the AI is totally lost.

```python
94:                 mem_reserved = 0
95:                 if torch.cuda.is_available():
96:                     mem_reserved = torch.cuda.max_memory_reserved() / (1024**2)
```
* **Lines 94 to 96 (The RAM Meter)**: **[Task 2 Change]** 
    *   **Line 95**: We ask: *"Do we have an Nvidia Graphics Card?"* 
    *   **Line 96**: This is a very deep command. `max_memory_reserved` asks your GPU: *"What was the absolutely heaviest you ever got during this entire run?"* We divide by `1024 / 1024` to convert the raw computer "bytes" into **Megabytes (MB)**—a number humans can easily read.

```python
98:                 print(f"Epoch {epoch+1} | Step {i} | Loss: {avg_train_loss:.4f} | PPL: {train_ppl:.2f} | Train Speed: {throughput:.2f} tok/s | Mem: {mem_reserved:.1f} MB")
```
* **Line 98 (The Status Report)**: We physically print all these variables into the dark black console window so you can watch your AI grow smarter and faster in real-time.

```python
100:                 total_loss = 0.0
101:                 start_time = time.time()
102:                 tokens_processed = 0
```
* **Lines 100 to 102 (The Reset)**: **This is critical!** After we print the update, we "Reset" the odometer and the stopwatch. If we didn't do this, our Speed calculation would get slower and slower over time because it would be averaging the entire day instead of just the last 5 minutes.

---

### 5. The Final Laboratory Report
Once the AI finishes reading the whole book (the Epoch), the Teacher runs one final, massive experiment.

```python
105:         val_loss, val_ppl, val_speed = evaluate(model, val_loader, device)
```
* **Line 105 (`evaluate`)**: **[Task 2 Change]** The Teacher takes the AI to a separate, private room and gives it a test it has never seen before. This function now returns not just the Grade (`val_ppl`), but also the **Inference Speed** (`val_speed`)—which tells you how fast the AI can "Write" after it's finished "Reading."

```python
107:         peak_mem = 0
108:         if torch.cuda.is_available():
109:             peak_mem = torch.cuda.max_memory_reserved() / (1024**2)
```
* **Lines 107 to 109 (`peak_mem`)**: One final check on the GPU's heart rate to see what the absolute heaviest memory load was during that validation test.

```python
111:         print(f"\n--- Experiment Results: {config.attention_type} ---")
112:         print(f"Validation PPL: {val_ppl:.2f}")
113:         print(f"Inference Speed: {val_speed:.2f} tokens/sec")
114:         print(f"Peak GPU Memory: {peak_mem:.1f} MB")
```
* **Lines 111 to 114 (The Results Block)**: **[Task 2 Change]** This is the formatted "Benchmarking Table" that appears at the very end of your run. This is exactly where you can see that **Linear Attention** (the variant we built) manages to stay at a very low memory number while having a high speed!

---

### 6. The "Main" Engine (Lines 117 & 118)
```python
117: if __name__ == '__main__':
118:     main()
```
* **Lines 117 & 118**: This is a standard Python rule. It says: *"If I physically double-click this file or type 'python train.py' in the terminal, then start the `main()` function."* It ensures that if another file tries to "borrow" a tool from this script, it doesn't accidentally start a full training run!
