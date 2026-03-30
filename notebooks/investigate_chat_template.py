"""
Kaggle notebook: Investigate the Nemotron-3-Nano chat template.
Run this on Kaggle with GPU to understand the exact prompt format.
Copy-paste this into a Kaggle notebook cell.
"""

# ============================================================
# CELL 1: Load tokenizer and inspect chat template
# ============================================================

import kagglehub
from transformers import AutoTokenizer

MODEL_PATH = kagglehub.model_download(
    "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("=" * 60)
print("CHAT TEMPLATE:")
print("=" * 60)
print(tokenizer.chat_template)

# ============================================================
# CELL 2: Test apply_chat_template with enable_thinking=True
# ============================================================

# This is exactly what the metric code does at inference time
user_content = (
    "In Alice's Wonderland, numbers are secretly converted into a different numeral system. "
    "Some examples are given below:\n"
    "11 -> XI\n15 -> XV\n94 -> XCIV\n19 -> XIX\n"
    "Now, write the number 38 in the Wonderland numeral system."
    "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
)

# Test 1: User message only (inference-style, what metric does)
prompt_inference = tokenizer.apply_chat_template(
    [{"role": "user", "content": user_content}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)

print("\n" + "=" * 60)
print("INFERENCE PROMPT (what metric sends to model):")
print("=" * 60)
print(repr(prompt_inference))
print()
print(prompt_inference)

# ============================================================
# CELL 3: Test training format (user + assistant)
# ============================================================

# Format 1: With thinking tags
assistant_with_thinking = (
    "<think>\n"
    "I need to convert 38 to Roman numerals.\n"
    "30 = XXX, 8 = VIII\n"
    "38 = XXXVIII\n"
    "</think>\n"
    "\\boxed{XXXVIII}"
)

# Format 2: Without thinking tags
assistant_no_thinking = "\\boxed{XXXVIII}"

for label, assistant_content in [
    ("WITH THINKING", assistant_with_thinking),
    ("WITHOUT THINKING", assistant_no_thinking),
]:
    try:
        formatted = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            tokenize=False,
            enable_thinking=True,
        )
        print(f"\n{'=' * 60}")
        print(f"TRAINING FORMAT ({label}):")
        print(f"{'=' * 60}")
        print(repr(formatted))
        print()
        print(formatted)
    except Exception as e:
        print(f"\nERROR with {label}: {e}")

# ============================================================
# CELL 4: Check special tokens
# ============================================================

print("\n" + "=" * 60)
print("SPECIAL TOKENS:")
print("=" * 60)
print(f"BOS: {tokenizer.bos_token!r} (id={tokenizer.bos_token_id})")
print(f"EOS: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
print(f"PAD: {tokenizer.pad_token!r} (id={getattr(tokenizer, 'pad_token_id', None)})")

# Check for thinking-related tokens
for name in ['think', 'thinking', 'thought']:
    matches = [t for t in tokenizer.get_vocab() if name in t.lower()]
    if matches:
        print(f"\nTokens containing '{name}': {matches[:10]}")

# ============================================================
# CELL 5: Tokenize and check lengths
# ============================================================

tokens = tokenizer.encode(prompt_inference)
print(f"\n{'=' * 60}")
print(f"INFERENCE PROMPT TOKEN COUNT: {len(tokens)}")
print(f"Max model context: 4096, Max generation: 3584")
print(f"Available for prompt: {4096 - 3584} = 512 tokens")
print(f"{'=' * 60}")

print("\nDONE - Copy the outputs above and share them back.")
