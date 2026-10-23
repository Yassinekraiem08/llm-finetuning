# LLM Fine-Tuning

A hands-on project exploring how fine-tuning can improve small language models, starting with **Qwen3-0.6B**.

The goal is to compare a pretrained model against a LoRA fine-tuned version on structured information extraction tasks.

### Progress
- Set up local inference using PyTorch and Apple Silicon GPU.
- Audited 47,649 examples from a Hugging Face JSON extraction dataset.
- Built a baseline evaluation pipeline.
- Achieved 100% parseable JSON after cleanup, but 0% exact-match accuracy on 10 samples.

### Next Steps
- Improve schema-guided evaluation.
- Fine-tune Qwen using LoRA.
- Compare results and publish the model on Hugging Face.

**Stack:** Python, PyTorch, Hugging Face Transformers, Datasets, PEFT.
