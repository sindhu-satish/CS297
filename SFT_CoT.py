from enum import Enum
from functools import partial
import pandas as pd
import torch

from transformers import AutoModelForCausalLM, TrainingArguments
from peft import get_peft_model, LoraConfig

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

import preprocess


model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
dataset_name = "sindhusatish97/MathQA"


train_dataset, test_dataset, tokenizer = preprocess.load_dataset_and_tokenizer(dataset_name, model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.resize_token_embeddings(len(tokenizer))

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

response_template = "<｜Assistant｜>"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


model = get_peft_model(model, peft_config)
for p in model.parameters():
    if not p.requires_grad:
        p.data = p.to(torch.float16)

output_dir = "cs297"
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
gradient_accumulation_steps = 1
logging_steps = 1
learning_rate = 1e-9
max_grad_norm = 0.3
max_steps = 2
num_train_epochs=1
warmup_ratio = 0.1
lr_scheduler_type = "cosine"
max_seq_length = 512

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    save_strategy="no",
    log_level="debug",
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    max_grad_norm=max_grad_norm,
    weight_decay=0.1,
    warmup_ratio=warmup_ratio,
    lr_scheduler_type=lr_scheduler_type,
    gradient_checkpointing=False,
    # bf16=False,
    bf16=False,
    fp16=True,
    report_to=["none"],
    hub_private_repo=False,
    push_to_hub=False,
    num_train_epochs=num_train_epochs,
    torch_empty_cache_steps=2,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    data_collator=collator
)

trainer.train()
trainer.save_model()