# Beginner's Guide to `data.py`

If `config.py` was our blueprint, `data.py` is our **Librarian**. Computers only understand math, not human words. The entire purpose of `data.py` is to download Wikipedia, correctly translate every single word into a unique math number ("Tokenization"), and string them together chronologically so the AI can read them.

We are going to go through every single line in order from top to bottom.

---

### Bringing in the Heavy Machinery (Lines 1 to 4)

```python
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import tiktoken
```
* **Line 1 (`import torch`)**: Brings in PyTorch, the massive library of math tools we use to build AI.
* **Line 2 (`from torch.utils...`)**: Brings in two specific PyTorch helper tools. `Dataset` is a tool for storing massive lists of data, and `DataLoader` is a tool (like a delivery truck) that knows how to grab chunks of that data quickly.
* **Line 3 (`from datasets...`)**: Brings in a tool from HuggingFace (an AI company) that knows how to automatically download Wikipedia from the internet for us.
* **Line 4 (`import tiktoken`)**: Brings in the exact dictionary created by OpenAI (the makers of ChatGPT) to translate words into numbers.

---

### Building the Flashcard Deck (Lines 6 to 20)

First, we design a custom "Librarian" whose only job is to properly hand the AI flashcards to study.

```python
class WikiTextDataset(Dataset):
```
* **Line 6**: This creates our custom Librarian class, and the `(Dataset)` part tells Python: *"This librarian is going to use the `Dataset` tool we imported on Line 2."*

```python
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
```
* **Lines 7 to 9**: This is the setup step for our Librarian. When the Librarian starts their shift, we hand them the giant list of millions of translated Wikipedia numbers (`data`), and we tell them exactly how many numbers they are allowed to put on a single flashcard (`seq_len`, which is 1024 from our blueprint). The Librarian explicitly memorizes these two things as `self.data` and `self.seq_len`.

```python
    def __len__(self):
        return len(self.data) // self.seq_len
```
* **Lines 11 & 12**: If someone asks the Librarian, *"Exactly how many 1024-word flashcards do you have?"*, this function calculates the answer. It asks for the total length of the massive dataset (`len(self.data)`) and divides it (`//`) by the length of a single flashcard (`self.seq_len`) to get the total number of perfectly-sized flashcards available.

```python
    def __getitem__(self, idx):
```
* **Line 14**: This is the most crucial part. This is the exact function that runs when the AI walks up to the Librarian and asks for a specific flashcard number (`idx`), for example, Flashcard #5.

```python
        start = idx * self.seq_len
        end = start + self.seq_len
        chunk = self.data[start:end+1]
```
* **Lines 15 to 17**: The Librarian calculates exactly where to cut the massive list. For flashcard #5, it multiplies `5 * 1024` to find the exact starting point (`start`), adds 1024 to find the end point (`end`), and cuts out that precise chunk of numbers. Note that it actually cuts out `end+1` (which means 1025 numbers long). Why 1025 instead of 1024? Because of the next step! 

```python
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
```
* **Lines 18 to 20**: Here we create the Student's Test (`x`) and the Teacher's Answer Key (`y`).
  * **Line 18 (`x`)**: `chunk[:-1]` says "give me everything in the list except the very last item." That turns the 1025-word list into a 1024-word Test.
  * **Line 19 (`y`)**: `chunk[1:]` says "skip the first item, and give me the rest of the list." That creates a 1024-word Answer Key that is shifted one word into the future.
  * **Line 20**: The Librarian hands both lists to the training loop.

#### 💡 WAIT, WHY DO WE EVEN DO THIS? (The Core Purpose of `x` and `y`)
This is actually the single most important concept in how AI "learns," called **Autoregressive Training**. 

If you show an AI an entire sentence at once ("The cat sat on the mat"), it learns nothing. To teach it how language *actually* flows, you must force it to play "Guess the Next Word" billions of times. But how do we automatically grade its guesses? 

We do it by shifting the sentence! Let's say the original sentence is: **"The cat sat on the mat"**

1. **`x` (The Student's Test) cuts off the last word:** "The cat sat on the"
2. **`y` (The Answer Key) cuts off the first word:** "cat sat on the mat"

Now, look at how perfectly they match up chronologically:
* The AI looks at Word #1 in `x` (**"The"**) and is asked to predict Word #1 in `y` (**"cat"**).
* The AI looks at Words #1 & #2 in `x` (**"The cat"**) and is asked to predict Word #2 in `y` (**"sat"**).
* The AI looks at Words #1, #2, & #3 in `x` (**"The cat sat"**) and predicts Word #3 in `y` (**"on"**).

By literally just shifting the same sentence by one word, we mathematically generated a thousands-of-questions-long exam with a perfect matching Answer Key! We hand both to the Teacher (`train.py`), who compares the AI's guesses to `y`, calculates how badly it failed, and forces the math to adapt.

---

### Connecting Everything Together (Lines 22 to 50)

Now that we define how the Librarian cuts out the flashcards, we need the actual data!

```python
def get_dataloaders(seq_len=1024, batch_size=16):
    print("Loading WikiText-2 dataset...")
```
* **Lines 22 & 23**: This defines the master function that our `train.py` script runs to kickstart everything. It takes in the `seq_len` and `batch_size` sizes from our blueprint, and prints out a message to the console.

```python
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
```
* **Lines 24 & 25**: We use that HuggingFace downloading tool from Line 3 to literally reach out to the internet, download millions of Wikipedia articles, and save them into a variable named `dataset`.

```python
    # Initialize tokenizer (using gpt2 tokenizer for basic BPE)
    enc = tiktoken.get_encoding("gpt2")
```
* **Lines 27 & 28**: We open OpenAI's official dictionary. This dictionary contains the exact math translations for 50,257 different words.

```python
    def encode_split(split):
        print(f"Tokenizing {split} split...")
        tokens = []
```
* **Lines 30 to 32**: This creates a custom mini-function block called `encode_split`. 
  * **Line 30 (`split`)**: The word `split` represents which part of the dataset we are looking at (like 'train' or 'test'). Why do we need a custom function instead of just doing it once? Because a good dataset is literally split into three different folders so we can test the AI fairly later! This function handles that generic logic.
  * **Line 31**: It announces to our terminal what it's currently doing so we don't think the computer froze.
  * **Line 32 (`tokens = []`)**: This creates a completely empty, blank list. We have to start with a blank list so that as we read the Wikipedia articles sentence-by-sentence next, we have a bucket to continuously dump the newly translated math numbers into.

```python
        for text in dataset[split]['text']:
            if text.strip():
                tokens.extend(enc.encode(text, allowed_special={"<|endoftext|>"}))
        return tokens
```
* **Lines 33 to 36**: This acts like a massive conveyor belt that processes Wikipedia one sentence at a time.
  * **Line 33 (`['text']`)**: When HuggingFace downloads a dataset, it organizes it like a massive Excel spreadsheet. It has multiple columns (Article ID, Author, Date, etc.). By putting `['text']` at the end, we tell Python to completely ignore the other metadata columns and only loop over the exact column that contains the actual English paragraphs.
  * **Line 34**: `if text.strip():` skips any rows in the spreadsheet that are completely blank, so we don't feed our AI useless silence.
  * **Line 35 (`enc.encode`)**: This is the heart of the translation! `enc.encode(text)` looks up every single English word in the OpenAI dictionary and translates them into math array numbers (like converting "Apple" into `4201`). 
    * **Wait, what is `allowed_special`?** OpenAI uses invisible, special "marker" words like `<|endoftext|>` to mathematically tell the AI exactly where one Wikipedia article finishes and a totally unrelated article begins. Normally, the dictionary crashes if it detects secret markers. By adding `allowed_special={"<|endoftext|>"}`, we give it explicit permission to translate that specific marker into a number so the AI can safely study it.
  * **Line 35 (`tokens.extend`)**: After translating a paragraph into math numbers, `extend` permanently jams those brand-new numbers into our `tokens[]` bucket from Line 32. 
  * **Line 36**: Once the conveyor belt finishes all millions of Wikipedia sentences, it returns the final, gigantic bucket of math numbers.

```python
    train_tokens = encode_split('train')
    val_tokens = encode_split('validation')
    test_tokens = encode_split('test')
```
* **Lines 38 to 40**: We run the translation function three times. Why? Because you never let an AI take a test on the exact same material it used to study! 
  * `train` is the textbook material it uses to learn grammar.
  * `validation` is a pop-quiz we give it mid-semester to gauge how well it is learning.
  * `test` is the final exam at the end of the year to prove it works.

```python
    train_ds = WikiTextDataset(train_tokens, seq_len)
    val_ds = WikiTextDataset(val_tokens, seq_len)
```
* **Lines 42 & 43**: We hire two Librarians (the class we built on Line 6). We give the first Librarian the textbook (`train_tokens`) and the second Librarian the pop-quiz (`val_tokens`). Both Librarians know to cut up their files into flashcards based on the `seq_len` (1024).

```python
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
```
* **Lines 45 & 46**: This creates our Delivery Trucks (`DataLoader`). If we just handed our thousands of flashcards directly to the mathematical AI brain one-by-one, it would be incredibly slow. Instead, PyTorch uses a `DataLoader` to violently optimize the supply chain.
  * **Line 45 (`DataLoader(train_ds...)`)**: This is the truck bringing study material from the `train` Librarian.
  * **Line 46 (`DataLoader(val_ds...)`)**: This is a separate truck bringing testing material from the `val` (pop-quiz) Librarian.
  * **What is `batch_size`?**: Remember in our blueprint we set `batch_size = 4`. Instead of carrying 1 flashcard into the AI, the truck stacks perfectly symmetrical blocks of 4 flashcards, and feeds them into the AI's math equations together simultaneously. This is massively faster because GPUs are designed to do parallel multiplication!
  * **Why `shuffle=True` vs `shuffle=False`?**: 
    * When studying (`train_loader`), we perfectly randomize the order of the flashcards. If we fed the AI Wikipedia chronologically, the AI might accidentally learn that "a paragraph about Rome usually follows a paragraph about Egypt" instead of actually learning English grammar! Shuffling breaks terrible contextual habits.
    * When taking an exam (`val_loader`), we set `shuffle=False` because the exam is just a metric. It doesn't matter what order we grade the test in; the AI isn't allowed to study from the test, so shuffling is mathematically irrelevant.

```python
    vocab_size = enc.n_vocab
    return train_loader, val_loader, vocab_size
```
* **Lines 48 & 49**: We mathematically confirm the dictionary size before completing our shift.
  * **What is `enc.n_vocab`?**: `enc` is the official OpenAI translation dictionary we stored on Line 28. `.n_vocab` (Number of Vocabulary) is a built-in command that simply asks that dictionary: *"Exactly how many total words do you physically have stored inside you?"* It replies with **`50257`**.
  * **Why do we need `vocab_size`?**: To the AI's math equations, `vocab_size` defines the number of unique "buttons" on its final output soundboard. When the AI guesses the next word, it doesn't write out English letters. It looks at a massive soundboard with exactly 50,257 buttons on it, calculates a percentage chance for every single button, and presses the one with the highest probability!
  * **Line 49**: The Librarian finishes the process by handing `train.py` the delivery truck full of training flashcards, the delivery truck full of pop-quiz flashcards, and the confirmed dictionary size!
