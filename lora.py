import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import get_peft_model, LoraConfig
from trl import SFTTrainer, SFTConfig
import wandb

# WandB 설정 초기화
wandb.init(project="lora_rank_experiment")

# 모델 및 토크나이저 설정
model_name = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 데이터셋 로드 및 토큰화
dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k")

def tokenize_function(examples):
    inputs = tokenizer(examples["instruction"], max_length=128, padding="max_length", truncation=True)
    outputs = tokenizer(examples["output"], max_length=128, padding="max_length", truncation=True)
    inputs["labels"] = outputs["input_ids"]
    return inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"]

# Data Collator 설정
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# LoRA rank 설정을 변경하면서 실험
for lora_r in [8, 128, 256]:
    print(f"\nStarting training with LoRA rank: {lora_r}")
    # LoRA 설정 구성
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=32,
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )

    # LoRA 모델 적용
    lora_model = get_peft_model(model, lora_config)

    # SFT 설정
    sft_config = SFTConfig(
        output_dir=f"/tmp/clm-instruction-tuning/rank_{lora_r}",
        max_seq_length=128,
        per_device_train_batch_size=8,
        num_train_epochs=1,  # 과제 진행을 위해 에폭을 적당히 조절
        logging_steps=50  # 로깅 빈도 조정
    )

    # Trainer 설정
    trainer = SFTTrainer(
        model=lora_model,
        train_dataset=train_dataset,
        args=sft_config,
        data_collator=collator
    )

    # 학습 시작 및 손실 로깅
    trainer.train()

    # 학습 손실 로깅 (WandB에 rank별로 구분하여 기록)
    for log in trainer.state.log_history:
        if "loss" in log:
            wandb.log({"train/loss": log["loss"], "train/global_step": log["step"], "rank": lora_r})

    # 메모리 사용량 출력 및 로깅
    max_memory_alloc = round(torch.cuda.max_memory_allocated(0) / 1024**3, 1)
    print(f"Max Allocated Memory for LoRA rank {lora_r}: {max_memory_alloc} GB")
    wandb.log({"lora_rank": lora_r, "max_memory_allocated": max_memory_alloc})

# 학습 종료 후 wandb 마무리
wandb.finish()
