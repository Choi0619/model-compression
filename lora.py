from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb
import torch
import time
from peft import get_peft_model, LoraConfig, TaskType

# 모델 및 토크나이저 설정
model_name = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 데이터셋 로드 및 전처리
dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train[:5%]")

# 텍스트 데이터 포맷을 문자열로 변환하는 함수 정의
def formatting_prompts_func(example):
    text = f"### Question: {example['instruction']}\n### Answer: {example['output']}"
    return {"text": text}

# 데이터셋에 전처리 적용
dataset = dataset.map(formatting_prompts_func)

# 데이터 콜레이터 설정
collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer,
    response_template="### Answer:"
)

# LoRA 설정을 위한 대상 모듈 선택
target_modules = set()
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        names = name.split(".")
        target_modules.add(names[0] if len(names) == 1 else names[-1])

if "lm_head" in target_modules:  # 필요한 경우 'lm_head' 제거
    target_modules.remove("lm_head")

target_modules = list(target_modules)

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
        dataset_kwargs={'skip_prepare_dataset': True}  # 데이터셋 준비 과정 건너뛰기
    )

    trainer = SFTTrainer(
        model=lora_model,
        train_dataset=dataset,
        args=sft_config,
        data_collator=collator,
    )

    # 학습 시작 시간 기록
    start_time = time.time()
    
    # 학습 시작
    trainer.train()
    
    # 학습 종료 시간 기록 및 계산
    end_time = time.time()
    epoch_runtime = end_time - start_time

    # WandB에 손실, 메모리 점유율, runtime 기록
    max_memory_alloc = round(torch.cuda.max_memory_allocated(0) / 1024**3, 1)
    for log in trainer.state.log_history:
        if "loss" in log:
            wandb.log({
                "train/loss": log["loss"],
                "train/global_step": log["step"],
                "rank": lora_r,
                "max_memory_allocated": max_memory_alloc,
                "train/grad_norm": log.get("grad_norm", None),
                "learning_rate": log.get("learning_rate", None),
                "runtime_per_epoch": epoch_runtime
            })

    # 최종 메모리 사용량과 학습 시간 출력
    print(f"Rank {lora_r} - Max Allocated Memory: {max_memory_alloc} GB - Runtime: {epoch_runtime:.2f}s")
    
    wandb.finish()
