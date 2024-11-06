import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb
import torch
from peft import get_peft_model, LoraConfig, TaskType

# 모델 및 토크나이저 설정
model_name = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 데이터셋 로드 및 전처리
dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train[:5%]")

# LoRA를 적용할 모듈을 지정하는 코드
target_modules = set()
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        names = name.split(".")
        target_modules.add(names[0] if len(names) == 1 else names[-1])

if "lm_head" in target_modules:  # 필요한 경우 'lm_head' 제거
    target_modules.remove("lm_head")

target_modules = list(target_modules)

# 학습용 데이터 포맷팅 함수 정의
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["instruction"])):
        text = f"### Question: {example['instruction'][i]}\n### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts

response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# LoRA rank 설정에 따른 학습 진행
for lora_r in [8, 128, 256]:
    torch.cuda.empty_cache()  # GPU 메모리 초기화
    wandb.init(project="LoRA_rank_experiment", name=f"LoRA rank {lora_r}", group="lora")

    # LoRA 설정
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,
    )
    lora_model = get_peft_model(model, lora_config)

    # SFTConfig 설정 및 트레이너 구성
    sft_config = SFTConfig(
        output_dir=f"/tmp/clm-instruction-tuning/rank_{lora_r}",
        max_seq_length=128,
        per_device_train_batch_size=8,
        num_train_epochs=1,
        logging_steps=10,
        learning_rate=5e-5,  # 초기 learning rate 설정
        lr_scheduler_type="cosine",  # 자동 learning rate 조정 스케줄러
        fp16=True,
    )

    trainer = SFTTrainer(
        model=lora_model,
        train_dataset=dataset,
        args=sft_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    # 학습 시작 시간 기록
    start_time = time.time()
    
    # 학습 시작
    trainer.train()
    
    # 학습 종료 시간 기록
    end_time = time.time()

    # 학습 소요 시간 계산 및 기록
    epoch_runtime = end_time - start_time
    wandb.log({"runtime_per_epoch": epoch_runtime, "lora_rank": lora_r})

    # WandB에 손실, 메모리 점유율 기록
    max_memory_alloc = round(torch.cuda.max_memory_allocated(0) / 1024**3, 1)
    print(f"Max Allocated Memory for LoRA rank {lora_r}: {max_memory_alloc} GB")
    for log in trainer.state.log_history:
        if "loss" in log:
            wandb.log({
                "train/loss": log["loss"],
                "train/global_step": log["step"],
                "rank": lora_r,
                "max_memory_allocated": max_memory_alloc,
                "train/grad_norm": log.get("grad_norm", None),
                "learning_rate": log.get("learning_rate", None)
            })

    wandb.finish()
