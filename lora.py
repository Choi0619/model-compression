from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb
import torch
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import time
import psutil
import os

# Load and split the dataset
dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train")
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset['train'].select(range(len(dataset['train']) // 20))  # Use 5% of the training data

model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-350m", 
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["instruction"])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts

response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# Get target modules
target_modules = set()
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        names = name.split(".")
        target_modules.add(names[0] if len(names) == 1 else names[-1])
if "lm_head" in target_modules:
    target_modules.remove("lm_head")
target_modules = list(target_modules)

for lora_r in [8, 128, 256]:
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    wandb.init(
        project="Hanghae99",
        name=f"rank_{lora_r}",
        group="lora",
        config={
            "lora_r": lora_r,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "learning_rate": 5e-4,
            "batch_size": 8,
            "max_seq_length": 128,
            "num_epochs": 5,
        }
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)

    initial_memory = get_memory_usage()
    initial_gpu_memory = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    wandb.log({
        "initial_cpu_memory_gb": initial_memory,
        "initial_gpu_memory_gb": initial_gpu_memory
    })

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=SFTConfig(
            output_dir=f"/tmp/lora_rank_{lora_r}",
            run_name=f"lora_rank_{lora_r}",  # Set a unique run name
            max_seq_length=128,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            fp16=True,
            logging_steps=10,
            learning_rate=5e-4,
            num_train_epochs=2,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            save_strategy="epoch",
            load_best_model_at_end=True,
        ),
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    start_time = time.time()
    train_result = trainer.train()
    end_time = time.time()
    
    final_memory = get_memory_usage()
    final_gpu_memory = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    runtime = end_time - start_time
    
    wandb.log({
        "final_cpu_memory_gb": final_memory,
        "final_gpu_memory_gb": final_gpu_memory,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
        "runtime_seconds": runtime,
        "runtime_minutes": runtime / 60,
        "final_train_loss": train_result.training_loss,
    })

    print(f"\nResults for LoRA rank {lora_r}:")
    print(f"Training Loss: {train_result.training_loss:.4f}")
    print(f"Runtime: {runtime/60:.2f} minutes")
    print(f"CPU Memory Usage: {final_memory:.2f} GB")
    print(f"GPU Memory Usage: {final_gpu_memory:.2f} GB")
    print(f"Peak GPU Memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print("-" * 50)

    wandb.finish()
