import pandas as pd
import datasets
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from transformers import TrainingArguments
import preprocess
from trl import SFTTrainer


max_seq_length = 32768 
dtype = None 
load_in_4bit = True 

dataset_name = "AI-MO/NuminaMath-CoT"
model_name = "unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit"


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit",
    max_seq_length = max_seq_length, trust_remote_code=True,
    dtype = dtype, 
    load_in_4bit = False)

model = FastLanguageModel.get_peft_model(
    model,
    r = 256,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 256,
    lora_dropout = 0.01, 
    bias = "none",   
    use_gradient_checkpointing = "unsloth",
    random_state = 42,
    use_rslora = True,  
    loftq_config = None,
)

train_dataset, test_dataset, tokenizer = preprocess.load_dataset_and_tokenizer(dataset_name, model_name)


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = ds_train,
    dataset_text_field = "text",
    max_seq_length = 5120,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        num_train_epochs = 4,
        learning_rate = 0.0001,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.08,
        lr_scheduler_type = "cosine",
        seed = 42,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

trainer_stats = trainer.train()
print("Trainer Stats: \n", trainer_stats)


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


trainer.save_model("outputs/sft_tir")
tokenizer.save_pretrained("outputs/sft_tir")