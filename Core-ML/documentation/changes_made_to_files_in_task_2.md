# Changes Made to Files in Task 2 - Line-by-Line Breakdown

We have made specific modifications to your baseline code to allow it to dynamically switch between the 3 new attention variants. Here is exactly what changed, line-by-line.

---

## ⚙️ 1. config.py
We added "Switches" so you can control the model without rewriting it.

- **Line 14 (`attention_type: str = "mqa"`)**: We added this variable so you can type `"standard"`, `"mqa"`, `"linear"`, or `"sliding_window"` to instantly change the AI's brain.
- **Line 15 (`window_size: int = 50`)**: This specifically controls how many words the **Sliding Window** variant looks at.

---

## 🧠 2. model.py
We made the model "Modular" so it can accept different types of attention.

- **Line 5 (`from attention_variants import ...`)**: We now import our new math engines from the separate file we created.
- **Line 22 (`class StandardAttention`)**: We renamed the original `Attention` class to `StandardAttention` so we can distinguish it from the new ones.
- **Lines 58-65 (Inside `TransformerBlock`)**: 
    - We added an `if/elif` statement. 
    - It looks at your `config.attention_type`.
    - If you typed `"mqa"`, it picks the `MultiQueryAttention` class.
    - If you typed `"linear"`, it picks `LinearAttention`.
- **Line 108 (`if self.config.attention_type != "linear"`)**: 
    - **This is Important!** Linear Attention handles "Causality" (not looking at the future) internally using a special math sum. 
    - Because it does it itself, we tell the model: "If we are using Linear Attention, don't waste time building an external mask."

---

## 📊 3. train.py
We turned your training script into a scientific benchmarking tool.

- **Line 27 (`start_time = time.time()`)**: We start a stopwatch right before the model starts "thinking" during validation.
- **Line 39 (`inference_throughput = ...`)**: We calculate your **Speed**. We take the total tokens processed and divide by the time it took. This tells us "Tokens per Second."
- **Lines 94-96 (`mem_reserved = ...`)**: We ask the GPU: "What is the absolute maximum amount of RAM you used during this training step?" and convert it to Megabytes (MB).
- **Line 105 (`val_loss, val_ppl, val_speed = evaluate(...)`)**: We updated the `evaluate` call to receive that new "Speed" metric we calculated.
- **Lines 111-115 (`print(...)`)**: We added a beautiful summary block that prints at the very end of your run, so you have all your numbers (PPL, Speed, Memory) in one place for your homework table.
