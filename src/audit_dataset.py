
import json
from collections import Counter

from datasets import load_dataset

# Load our dataset
dataset = load_dataset("robdixon/json-extraction")

for split_name in ["train", "validation"]:
    split = dataset[split_name]
    total = len(split)

    valid_json = 0
    invalid_json = 0
    empty_inputs = 0
    example_pairs = []

    for example in split:
        # Check JSON syntax
        try:
            json.loads(example["json"])
            valid_json += 1
        except (json.JSONDecodeError, TypeError):
            invalid_json += 1

        # Check missing or empty inputs
        instruction = example["instruction"] or ""
        text = example["text"] or ""

        if not instruction.strip() or not text.strip():
            empty_inputs += 1

        # Identify identical input pairs
        example_pairs.append((instruction, text))

    counts = Counter(example_pairs)
    duplicate_rows = sum(count - 1 for count in counts.values())

    print(f"\n--- {split_name.upper()} ---")
    print(f"Total examples: {total}")
    print(f"Valid JSON: {valid_json}")
    print(f"Invalid JSON: {invalid_json}")
    print(f"JSON validity: {valid_json / total:.2%}")
    print(f"Empty inputs: {empty_inputs}")
    print(f"Duplicate input rows: {duplicate_rows}")


# Check overlap between training and validation inputs
train_inputs = {
    (example["instruction"], example["text"])
    for example in dataset["train"]
}

validation_inputs = {
    (example["instruction"], example["text"])
    for example in dataset["validation"]
}

overlap = train_inputs & validation_inputs

print("\n--- DATA LEAKAGE CHECK ---")
print(f"Overlapping input pairs: {len(overlap)}")
print(f"Validation overlap rate: {len(overlap) / len(validation_inputs):.2%}")
