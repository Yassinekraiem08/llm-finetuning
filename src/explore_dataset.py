
from datasets import load_dataset

dataset = load_dataset("robdixon/json-extraction")

print("Available splits:")
print(dataset)

print("\nColumns:")
for split in dataset:
    print(split, dataset[split].column_names)

print("\nFirst example:")
print(dataset[list(dataset.keys())[0]][0])


import json

print("\n--- TRAINING EXAMPLES ---")

for i in range(5):
    example = dataset["train"][i]

    print(f"\nExample {i + 1}")
    print("Instruction:", example["instruction"])
    print("Text:", example["text"][:500])
    print("Expected JSON:", example["json"])

    try:
        json.loads(example["json"])
        print("Valid JSON: YES")
    except json.JSONDecodeError:
        print("Valid JSON: NO")
