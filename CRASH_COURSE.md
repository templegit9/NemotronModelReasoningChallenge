# Crash Course: What We're Doing

## The Competition

We're in a **Kaggle competition** by NVIDIA worth **$106,388 in prizes**. The deadline is **June 15, 2026**.

The goal: make an AI model better at solving reasoning puzzles.

---

## The Model

NVIDIA gave everyone the same AI model to work with: **Nemotron-3-Nano-30B-A3B**.

Think of it like a student who already knows a lot (30 billion parameters of knowledge), but we need to teach it how to solve specific types of puzzles. We can't change the student's brain — we can only give it a small "cheat sheet" to carry into the exam.

That cheat sheet is called a **LoRA adapter**.

### What is LoRA?

LoRA (Low-Rank Adaptation) is a technique for fine-tuning (customizing) a large AI model **without modifying the original model**. Instead of retraining all 30 billion parameters (which would require enormous compute), LoRA adds tiny trainable layers on top — like sticky notes on a textbook.

Key constraints set by the competition:
- **Rank <= 32**: This limits how big our "sticky notes" can be. Higher rank = more expressive but more memory.
- Our adapter is ~880 million trainable params out of 32 billion total (2.7%).

### What is Mamba?

This isn't a normal Transformer model (like ChatGPT). It's a **hybrid** that mixes:
- **Transformer layers** (the standard attention-based architecture)
- **Mamba layers** (a newer architecture called a State Space Model / SSM)

Mamba is faster for long sequences but requires special CUDA kernels (GPU code) to run, which is why we had to deal with all those `ptxas` permission errors.

---

## The Puzzles (6 Categories)

The training data has **9,500 puzzles** equally split across 6 types. Every puzzle is framed as "In Alice's Wonderland..." and gives examples, then asks the model to solve a new one.

### 1. Number System (Roman Numerals)
> "11 -> XI, 15 -> XV, 94 -> XCIV. Now write 38."
>
> Answer: XXXVIII

Convert decimal numbers to Roman numerals. Straightforward rules.

### 2. Unit Conversion
> "1.0m becomes 3.281, 2.0m becomes 6.562. Convert 5.0m."
>
> Answer: 16.405

Find the conversion factor from examples, then multiply.

### 3. Physics (Free Fall)
> "At t=1s, distance=4.9m. At t=2s, distance=19.6m. Find distance at t=3s."
>
> Answer: 44.1

Uses `distance = 0.5 x g x time^2`. Solve for gravity from examples, then compute.

### 4. Text Encryption (Substitution Cipher)
> "abc -> xyz, def -> wvu. Decrypt: xyw."
>
> Answer: abd

Each letter maps to another letter. Build the mapping from examples, apply it.

### 5. Bit Manipulation
> "10110010 -> 01001101, 11110000 -> 00001111. Find output for: 10101010"

Binary string transformations. Could be XOR, bit reversal, shifts, or combinations. **Very hard** — many possible rules.

### 6. Symbol Transform
> "ABC -> XYZ, DEF -> UVW. Find result for: GHI"

Character-level substitution or transformation rules. Also **very hard**.

---

## Our Strategy

### Step 1: Build Solvers (Done)

We wrote Python programs that can solve each puzzle type algorithmically:

| Category | Our Solver Accuracy |
|----------|-------------------|
| Number System | 100% |
| Unit Conversion | 100% |
| Physics | 100% |
| Text Encryption | 99.9% |
| Bit Manipulation | 8.7% |
| Symbol Transform | 0.7% |

The easy categories are fully solved. The hard ones (bit/symbol) have too many possible rules for a simple solver.

### Step 2: Generate Training Data (Done)

We used our solvers to create **chain-of-thought (CoT) reasoning** for each puzzle. Instead of just giving the model the answer, we show it *how to think*:

```
<think>
I need to convert 38 to Roman numerals.
Breaking it down: 10 x 3 = XXX, 5 x 1 = V, 1 x 3 = III
So 38 = XXXVIII
</think>
\boxed{XXXVIII}
```

This teaches the model to reason step-by-step before answering, which dramatically improves accuracy.

### Step 3: Fine-Tune on Kaggle (In Progress)

We train the LoRA adapter on Kaggle's GPU (NVIDIA RTX Pro 6000, 48GB VRAM). The training:

1. Shows the model a puzzle
2. Shows it the correct chain-of-thought reasoning + answer
3. Adjusts the LoRA weights to make the model more likely to produce that reasoning
4. Repeats for all 8,553 training examples, twice (2 epochs)

**Loss** is the number we watch during training. Lower = better. It measures how surprised the model is by the correct answer. We saw it drop from 1.12 to 1.04 in the first 450 steps — the model is learning.

### Step 4: Submit and Score

After training, we save the LoRA adapter and submit it. Kaggle's evaluation:

1. Loads the base Nemotron model + our adapter
2. Feeds it test puzzles (ones we've never seen)
3. The model generates reasoning + answer
4. Scores accuracy: did it get the right answer?

**Important**: Evaluation uses `temperature=1.0`, which adds randomness to the model's output. This means scores vary between runs — the same adapter can score differently each time.

### Step 5: Iterate and Improve (Upcoming)

We plan to:
- **Generate synthetic data**: Create more training puzzles (especially for hard categories)
- **Use autoresearch**: An automated experiment loop (by Andrej Karpathy) that tries different training configurations overnight and keeps the best ones
- **Optimize per-category**: Tune the training to focus more on categories where we're weakest

---

## Key Concepts Glossary

| Term | What It Means |
|------|--------------|
| **Fine-tuning** | Customizing a pre-trained model for a specific task |
| **LoRA** | A lightweight fine-tuning method that adds small trainable layers |
| **Adapter** | The small set of weights we train and submit (our "cheat sheet") |
| **Chain-of-Thought (CoT)** | Teaching the model to show its reasoning step-by-step |
| **Epoch** | One complete pass through all training data |
| **Batch size** | How many examples the model sees at once (we use 2) |
| **Gradient accumulation** | Simulating a larger batch by accumulating gradients over multiple small batches (we use 8, so effective batch = 16) |
| **Learning rate** | How big of a step the model takes when adjusting weights (too high = overshoots, too low = learns too slowly) |
| **Loss** | How wrong the model's predictions are (lower = better) |
| **bf16 / bfloat16** | A number format that uses half the memory of normal floats, with minimal accuracy loss |
| **VRAM** | Video RAM — the GPU's memory. Our model needs ~12GB, and we have 48GB on Kaggle |
| **vLLM** | The inference engine Kaggle uses to run our model during evaluation |
| **Tokenizer** | Converts text to numbers (tokens) that the model can process |
| **Chat template** | The exact format for structuring conversations with the model (system/user/assistant messages) |
| **\boxed{}** | LaTeX-style formatting the competition uses to extract the final answer |
| **CUTLASS** | NVIDIA's GPU kernel library, required for Mamba layers |
| **ptxas** | NVIDIA's PTX assembler — compiles GPU code. The permission errors we fought were about this binary being on a read-only filesystem |

---

## File Structure

```
NemotronModelReasoningChallenge/
  data/
    train.csv              # Original 9,500 competition examples
    train_split.csv        # 8,553 training examples (90%)
    val_split.csv          # 947 validation examples (10%)
    train_formatted.jsonl  # Training data with CoT reasoning
    val_formatted.jsonl    # Validation data with CoT reasoning
  solvers/                 # Algorithmic solvers for each puzzle type
    roman_numeral.py
    unit_conversion.py
    physics.py
    caesar_cipher.py       # Actually monoalphabetic substitution, not Caesar
    bit_manipulation.py
    symbol_transform.py
  notebooks/
    train_submission.py    # The Kaggle training notebook (currently running)
    investigate_chat_template.py
    submission_demo.ipynb  # Official starter notebook
    metric.ipynb           # Official evaluation code
  categorize_data.py       # Classifies puzzles into 6 categories
  generate_training_data.py # Creates CoT-formatted JSONL
  verify_solvers.py        # Tests solver accuracy against training data
  TASKS.md                 # Project roadmap and task tracker
  CRASH_COURSE.md          # This file
```

---

## Where We Are Right Now

Training is running on Kaggle (~5-6 hours total). When it finishes:

1. It saves the LoRA adapter
2. We click "Submit to competition"
3. We get our first real score
4. Then we start optimizing
