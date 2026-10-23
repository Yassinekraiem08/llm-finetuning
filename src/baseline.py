
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Choose our pretrained language model
model_name = "Qwen/Qwen3-0.6B"

# 2. Select Apple's GPU
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {device}")

# 3. Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 4. Load the pretrained model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16 if device == "mps" else torch.float32,
).to(device)

model.eval()

# 5. Define our prompt
messages = [
    {
        "role": "user",
        "content": "Explain what fine-tuning means in machine learning in two sentences."
    }
]

# 6. Convert the conversation into model inputs
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_dict=True,
    return_tensors="pt",
).to(device)

# 7. Generate a response
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

# 8. Decode only the newly generated tokens
generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print("\nMODEL RESPONSE:\n")
print(response)