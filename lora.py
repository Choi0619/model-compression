from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb
import torch
from peft import get_peft_model, LoraConfig, TaskType
import time
import psutil
import os

# 데이터셋 로드 및 분할
dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train")
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset['train'].select(range(len(dataset['train']) // 20))  # 학습 데이터의 5% 사용

# 모델과 토크나이저 로드
model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-350m", 
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

# 메모리 사용량을 계산하는 함수
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024  # GB 단위로 변환

# 프롬프트 포맷팅 함수
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["instruction"])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts

response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# 타겟 모듈 선택
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

    # 초기 메모리 사용량 기록
    initial_memory = get_memory_usage()
    initial_gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
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
        ),
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )


    start_time = time.time()
    train_result = trainer.train()
    end_time = time.time()
    
    # 평균 메모리 사용량 계산
    final_memory = get_memory_usage()
    final_gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    avg_cpu_memory = (initial_memory + final_memory) / 2
    avg_gpu_memory = (initial_gpu_memory + final_gpu_memory) / 2
    runtime = end_time - start_time
    
    wandb.log({
        "avg_cpu_memory_gb": avg_cpu_memory,
        "avg_gpu_memory_gb": avg_gpu_memory,
        "runtime_seconds": runtime,
        "runtime_minutes": runtime / 60,
        "final_train_loss": train_result.training_loss,
    })

    print(f"\nResults for LoRA rank {lora_r}:")
    print(f"Training Loss: {train_result.training_loss:.4f}")
    print(f"Runtime: {runtime/60:.2f} minutes")
    print(f"Average CPU Memory Usage: {avg_cpu_memory:.2f} GB")
    print(f"Average GPU Memory Usage: {avg_gpu_memory:.2f} GB")
    print("-" * 50)

    wandb.finish()
