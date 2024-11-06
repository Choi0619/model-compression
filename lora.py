import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import wandb
import torch.nn.functional as F

# WandB 설정 초기화
wandb.init(project="lora_rank_experiment")

# 모델 및 토크나이저 설정
model_name = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 데이터셋 로드 및 토큰화
dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k")

def tokenize_function(examples):
    # 'instruction'과 'output' 열을 사용하여 입력 시퀀스를 생성하고, 라벨을 'labels'에 설정
    inputs = tokenizer(
        examples["instruction"], padding="max_length", truncation=True, max_length=128
    )
    labels = tokenizer(
        examples["output"], padding="max_length", truncation=True, max_length=128
    )

    # 'labels' 필드를 추가하여 모델이 loss를 계산할 수 있도록 합니다.
    inputs["labels"] = labels["input_ids"]
    return inputs

# 토큰화된 데이터셋을 생성합니다.
tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"]

# LoRA 적용할 모듈 설정
target_modules = set()
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):  # 모델에서 torch.nn.Linear에만 LoRA 적용
        names = name.split('.')
        target_modules.add(names[0] if len(names) == 1 else names[-1])
if "lm_head" in target_modules:  # 필요시 lm_head 제거
    target_modules.remove("lm_head")
target_modules = list(target_modules)

# 커스텀 Trainer 정의
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 모델 출력과 라벨을 받아서 손실을 계산
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # 레이블을 가져와서 손실 계산
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("labels key is missing in inputs")
        
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# 각 lora_r 값에 대해 실험
for lora_r in [8, 128, 256]:
    print(f"\nStarting training with LoRA rank: {lora_r}")
    
    # LoRA 설정 구성
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=32,
        lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM,  # Causal Language Modeling
        target_modules=target_modules
    )

    # LoRA 모델 적용
    lora_model = get_peft_model(model, lora_config)

    # TrainingArguments 설정
    training_args = TrainingArguments(
        output_dir=f"/tmp/clm-instruction-tuning/rank_{lora_r}",
        per_device_train_batch_size=8,
        num_train_epochs=1,
        logging_steps=10,
        report_to="wandb"  # wandb에 로그 기록
    )

    # CustomTrainer 사용
    trainer = CustomTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )

    # 학습 시작 및 손실 로깅
    trainer.train()

    # 학습 종료 후 메모리 사용량 출력
    max_memory_alloc = round(torch.cuda.max_memory_allocated(0) / 1024**3, 1)
    print(f"Max Allocated Memory for LoRA rank {lora_r}: {max_memory_alloc} GB")
    wandb.log({"lora_rank": lora_r, "max_memory_allocated": max_memory_alloc})

# 학습 종료 후 wandb 마무리
wandb.finish()
