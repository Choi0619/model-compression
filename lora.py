from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb
import torch
from peft import get_peft_model, LoraConfig, TaskType

# LoRA 설정 변수
lora_r: int = 8
lora_dropout: float = 0.1
lora_alpha: int = 32

# 모델 및 토크나이저 설정
model_name = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 데이터셋 로드
dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train[:17%]")

# LoRA를 적용할 모듈 선택
target_modules = set()
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        names = name.split(".")
        target_modules.add(names[0] if len(names) == 1 else names[-1])

if "lm_head" in target_modules:
    target_modules.remove("lm_head")

target_modules = list(target_modules)

# 학습용 데이터 포맷팅 함수 정의
def formatting_prompts_func(example):
    return f"### Question: {example['instruction']}\n### Answer: {example['output']}"

response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# LoRA rank 설정에 따른 학습 진행
for lora_r in [8, 128, 256]:
    torch.cuda.empty_cache()
    wandb.init(project="Hanghae99", name=f"rank {lora_r}", group="lora")

    # LoRA 설정
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    lora_model = get_peft_model(model, lora_config)

    # SFTConfig 설정
    sft_config = SFTConfig(
        output_dir=f"/tmp/clm-instruction-tuning/rank_{lora_r}",
        max_seq_length=128,
        per_device_train_batch_size=16,
        fp16=True,
        logging_steps=1,
        remove_unused_columns=False  # 불필요한 컬럼 제거 옵션 비활성화
    )

    trainer = SFTTrainer(
        model=lora_model,
        train_dataset=dataset,
        args=sft_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    # 학습 시작
    trainer.train()

    # 학습 종료 후 메모리 사용량 출력
    max_memory_alloc = round(torch.cuda.max_memory_allocated(0) / 1024**3, 1)
    print(f"Rank {lora_r} - Max Allocated Memory: {max_memory_alloc} GB")

    # WandB 종료
    wandb.finish()
