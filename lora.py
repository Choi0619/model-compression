from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb
import torch
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import time
import psutil
import os

# Increase dataset size to 20%
dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train[:20%]")
model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-350m", 
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

def get_memory_usage():
    """Get current memory usage of the process"""
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
    # Clear CUDA cache and reset model
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Initialize wandb with more detailed config
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

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)

    # Log initial memory usage
    initial_memory = get_memory_usage()
    initial_gpu_memory = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    wandb.log({
        "initial_cpu_memory_gb": initial_memory,
        "initial_gpu_memory_gb": initial_gpu_memory
    })

    # Configure trainer with improved parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=f"/tmp/lora_rank_{lora_r}",
            max_seq_length=128,
            per_device_train_batch_size=8,  # Reduced batch size
            gradient_accumulation_steps=2,   # Added gradient accumulation
            fp16=True,
            logging_steps=10,               # Increased logging frequency
            learning_rate=5e-4,            # Increased learning rate
            num_train_epochs=5,            # Increased epochs
            warmup_ratio=0.1,              # Increased warmup
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,             # Added gradient clipping
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
        ),
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    # Training with timing
    start_time = time.time()
    train_result = trainer.train()
    end_time = time.time()
    
    # Log final metrics
    final_memory = get_memory_usage()
    final_gpu_memory = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    runtime = end_time - start_time
    
    wandb.log({
        "final_cpu_memory_gb": final_memory,
        "final_gpu_memory_gb": final_gpu_memory,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
        "runtime_seconds": runtime,
        "runtime_minutes": runtime / 60,
        "final_loss": train_result.training_loss,
    })

    # Print detailed metrics
    print(f"\nResults for LoRA rank {lora_r}:")
    print(f"Training Loss: {train_result.training_loss:.4f}")
    print(f"Runtime: {runtime/60:.2f} minutes")
    print(f"CPU Memory Usage: {final_memory:.2f} GB")
    print(f"GPU Memory Usage: {final_gpu_memory:.2f} GB")
    print(f"Peak GPU Memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print("-" * 50)

    wandb.finish()