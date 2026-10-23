
import json
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------------------
# Configuration
# ----------------------------------------

MODEL_NAME = "Qwen/Qwen3-0.6B"
DATASET_NAME = "robdixon/json-extraction"
NUM_EXAMPLES = 10
SEED = 42

device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {device}")


# ----------------------------------------
# 1. JSON parsing helper
# ----------------------------------------

def parse_json_response(response):
    """
    Parse JSON after removing optional
    Markdown code fences.
    """
    response = response.strip()

    if response.startswith("```"):
        lines = response.splitlines()

        if len(lines) >= 3 and lines[-1].strip() == "```":
            response = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return None


# ----------------------------------------
# 2. Load pretrained model
# ----------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if device == "mps" else torch.float32,
).to(device)

model.eval()


# ----------------------------------------
# 3. Load validation dataset
# ----------------------------------------

dataset = load_dataset(
    DATASET_NAME,
    split="validation",
)

samples = dataset.shuffle(seed=SEED).select(
    range(NUM_EXAMPLES)
)


# ----------------------------------------
# 4. Initialize evaluation metrics
# ----------------------------------------

raw_valid_count = 0
cleaned_valid_count = 0
exact_count = 0


# ----------------------------------------
# 5. Evaluate model
# ----------------------------------------

for i, example in enumerate(samples):

    prompt = (
        f"{example['instruction']}\n\n"
        f"Text:\n{example['text']}\n\n"
        "Return only valid JSON. Do not add explanations."
    )

    messages = [
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][
        inputs["input_ids"].shape[-1]:
    ]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    # ------------------------------------
    # 6. Raw JSON validity
    # ------------------------------------

    raw_valid = False

    try:
        json.loads(response)
        raw_valid = True
        raw_valid_count += 1
    except json.JSONDecodeError:
        pass

    # ------------------------------------
    # 7. Cleaned JSON validity
    # ------------------------------------

    prediction = parse_json_response(response)

    cleaned_valid = prediction is not None

    if cleaned_valid:
        cleaned_valid_count += 1

    # ------------------------------------
    # 8. Exact match accuracy
    # ------------------------------------

    expected = json.loads(example["json"])

    exact = cleaned_valid and prediction == expected

    if exact:
        exact_count += 1

    # ------------------------------------
    # 9. Print individual results
    # ------------------------------------

    print(f"\n--- EXAMPLE {i + 1} ---")

    print(f"Raw JSON valid: {raw_valid}")
    print(f"Cleaned JSON valid: {cleaned_valid}")
    print(f"Exact match: {exact}")


# ----------------------------------------
# 10. Final evaluation results
# ----------------------------------------

print("\n" + "=" * 50)
print("BASELINE EVALUATION RESULTS")
print("=" * 50)

print(f"Model: {MODEL_NAME}")
print(f"Examples evaluated: {NUM_EXAMPLES}")

print(
    f"Raw JSON validity: "
    f"{raw_valid_count / NUM_EXAMPLES:.2%}"
)

print(
    f"Cleaned JSON validity: "
    f"{cleaned_valid_count / NUM_EXAMPLES:.2%}"
)

print(
    f"Exact match accuracy: "
    f"{exact_count / NUM_EXAMPLES:.2%}"
)

print("=" * 50)
