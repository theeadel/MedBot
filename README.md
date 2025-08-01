# 🧬 MedBot: Fine-Tuned LLM on Medical QA Data

MedBot is a lightweight medical question-answering chatbot built by fine-tuning a quantized LLM (GPT-NeoX-20B) using LoRA on the `medalpaca/medical_meadow_wikidoc_patient_information` dataset. The project demonstrates how to customize large language models efficiently for domain-specific tasks using Hugging Face + PEFT.

---

## 🚀 Features

- 🔎 Fine-tuned EleutherAI/GPT-NeoX-20B using 4-bit quantization
- 🧠 PEFT with LoRA to reduce training overhead
- 📚 Medical QA dataset from Hugging Face (MedAlpaca)
- ⚡ Mixed-precision + gradient checkpointing for low-memory training
- 🧪 Custom prompt-based inference

---

## 🧱 Project Structure

```
medbot-lora/
├── medbot_train.py              # Main fine-tuning script
├── outputs/                     # Saved fine-tuned model
├── logs/                        # Training logs
├── README.md                    # Project overview
```

---

## 📦 Requirements

Install the dependencies:
```bash
pip install bitsandbytes
pip install git+https://github.com/huggingface/transformers.git
pip install git+https://github.com/huggingface/peft.git
pip install git+https://github.com/huggingface/accelerate.git
pip install datasets
```

---

## 📊 Dataset

We use the [MedAlpaca - WikiDoc Patient Info](https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc_patient_information) dataset via:
```python
from datasets import load_dataset

data = load_dataset("medalpaca/medical_meadow_wikidoc_patient_information")
```

---

## 🧠 Training Configuration

Key parameters:
- `per_device_train_batch_size = 8`
- `gradient_accumulation_steps = 4`
- `fp16 = True`
- `optim = "adamw_bnb_8bit"`

Trainer setup:
```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
```

---

## 💬 Inference Example

```python
prompt = "What is the capital of France?"
response = model.generate(...)
```

Example output:
```
The capital of France is Paris.
```

---

## 💾 Model Output

After training, save your LoRA-tuned model using:
```python
model.save_pretrained("outputs")
```

---

## 📌 Future Improvements
- Add a frontend using Streamlit or Gradio
- Integrate a chatbot UI with Flask or FastAPI
- Expand dataset to include other domains (legal, finance, etc.)
- Support instruction tuning and prompt chaining

---

## 🙏 Credits
- Developed by [Zaka AI](https://zaka.ai/)
- Dataset by [MedAlpaca](https://huggingface.co/datasets/medalpaca)

> A small step toward open-source medical AI 🩺
