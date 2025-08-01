# Build Your MedBot with LoRA Fine-Tuning on Medical Data

# =============================
# 📦 Install Required Libraries
# =============================
!pip install bitsandbytes
!pip install git+https://github.com/huggingface/transformers.git
!pip install git+https://github.com/huggingface/peft.git
!pip install git+https://github.com/huggingface/accelerate.git
!pip install datasets

# =============================
# 🔁 Imports
# =============================
import torch
import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import re

# =============================
# 📥 Load Base Model and Tokenizer (4bit)
# =============================
hf_model = "EleutherAI/gpt-neox-20b"

bitsbytes_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(hf_model)
model = AutoModelForCausalLM.from_pretrained(
    hf_model,
    quantization_config=bitsbytes_config,
    device_map="auto"
)

# =============================
# 🧪 Prepare for Training
# =============================
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# LoRA Configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Print trainable parameters
def print_trainable_parameters(model):
    trainable, total = 0, 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    print(f"Trainable: {trainable} || Total: {total} || %: {100 * trainable / total:.2f}%")

print_trainable_parameters(model)

# =============================
# 📊 Load and Preprocess Dataset
# =============================
data = load_dataset("medalpaca/medical_meadow_wikidoc_patient_information")
data = data["train"].train_test_split(test_size=0.2, seed=42)

tokenizer.pad_token = tokenizer.eos_token

# Tokenize input field
data = data.map(lambda samples: tokenizer(samples["input"], padding="max_length", truncation=True), batched=True)

# =============================
# 🧠 Training Configuration
# =============================
training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    logging_dir="./logs",
    logging_steps=500,
    fp16=True,
    optim="adamw_bnb_8bit",
    dataloader_num_workers=2,
    ddp_find_unused_parameters=False,
    report_to="none",
)

model.config.use_cache = False

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

# =============================
# 💾 Save Trained Model
# =============================
saved_model = model if hasattr(model, "save_pretrained") else model.module
saved_model.save_pretrained("outputs")

# =============================
# 🔁 Load LoRA Configs (for inference only)
# =============================
reload_lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=1,
    lora_alpha=1,
    lora_dropout=0.05
)

model = get_peft_model(model, reload_lora_config)

# =============================
# 💬 Inference: Ask MedBot
# =============================
prompt = "What is the capital of France?"
device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(
    **inputs,
    max_new_tokens=40,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.3
)

decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
cleaned_text = re.sub(r'<[^>]+>', '', decoded_text)
print(cleaned_text)
