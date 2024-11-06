import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb
from peft import get_peft_model, LoraConfig, TaskType

# WandB 설정 초기화
wandb.init(project="LoRA_rank_experiment")

# 모델 및 토크나이저 설정
model_name = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 데이터셋 로드 및 전처리
dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train[:17%]")

def formatting_prompts_func(example):
    # 데이터셋의 instruction과 output을 포맷팅하여 텍스트 형식으로 변환
    output_texts = []
    for i in range(len(example["instruction"])):
        text = f"### Question: {example['instruction'][i]}\n### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts

# LoRA를 적용할 모듈을 지정하는 코드
target_modules = set()
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        names = name.split(".")
        target_modules.add(names[0] if len(names) == 1 else names[-1])

if "lm_head" in target_modules:  # 필요한 경우 'lm_head' 제거
    target_modules.remove("lm_head")

target_modules = list(target_modules)
response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# 각 rank에 대한 학습 반복
for lora_r in [8, 128, 256]:
    print(f"\nStarting training with LoRA rank: {lora_r}")
    torch.cuda.empty_cache()  # GPU 메모리 초기화

    # LoRA 설정
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,
    )

    # LoRA 모델 적용
    lora_model = get_peft_model(model, peft_config)

    # SFTConfig 설정 및 트레이너 구성
    sft_config = SFTConfig(
        output_dir=f"/tmp/clm-instruction-tuning/rank_{lora_r}",
        max_seq_length=128,
        per_device_train_batch_size=8,
        num_train_epochs=1,
        logging_steps=50,
        fp16=True,
    )

    trainer = SFTTrainer(
        model=lora_model,
        train_dataset=dataset,
        args=sft_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    # 학습 시작 및 WandB 로그 기록
    trainer.train()

    # 메모리 사용량 및 학습 손실 기록
    max_memory_alloc = round(torch.cuda.max_memory_allocated(0) / 1024**3, 1)
    print(f"Max Allocated Memory for LoRA rank {lora_r}: {max_memory_alloc} GB")
    for log in trainer.state.log_history:
        if "loss" in log:
            wandb.log({
                "train/loss": log["loss"],
                "train/global_step": log["step"],
                "rank": lora_r,
                "max_memory_allocated": max_memory_alloc
            })

    # Rank 학습 종료 후 WandB 세션 종료
    wandb.finish()
