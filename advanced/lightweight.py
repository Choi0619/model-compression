import json
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, BitsAndBytesConfig
import wandb
import time
import psutil
import torch
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

# WandB 초기화
wandb.init(project="therapist-chatbot", name="gpt-neo-1.3B-training-quantized")

# corpus.json 데이터 로드
with open('corpus.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

# 입력-출력 쌍 준비
data_pairs = []
for i in range(0, len(corpus)-1, 2):  # user와 therapist 쌍으로 진행
    if corpus[i]['role'] == 'user' and corpus[i+1]['role'] == 'therapist':
        input_text = corpus[i]['content']  # 사용자 입력
        output_text = corpus[i + 1]['content']  # 치료사 응답
        data_pairs.append({"input": input_text, "output": output_text})

# 전체 데이터를 학습 세트로 사용
train_data = data_pairs

# Hugging Face 데이터셋으로 변환
train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))

# 모델과 토크나이저 로드
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

# pad_token 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 모델의 모듈 탐색 (모듈 이름 확인)
for name, module in model.named_modules():
    print(name)

# LoRA 적용 가능한 모듈로 설정
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,  # rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]  # gpt-neo-1.3B의 적합한 모듈 예시
)

# LoRA 적용
model = get_peft_model(model, lora_config)

# 전처리 함수 정의
def preprocess_function(examples):
    inputs = tokenizer(examples['input'], max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(text_target=examples['output'], max_length=256, truncation=True, padding="max_length").input_ids
    
    # <pad> 토큰을 -100으로 설정하여 손실 계산에서 제외
    labels = [[(label if label != tokenizer.pad_token_id else -100) for label in label_list] for label_list in labels]
    
    inputs["labels"] = labels
    return inputs

# 전처리 적용
train_dataset = train_dataset.map(preprocess_function, batched=True)

# DataCollatorWithPadding 사용
collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    logging_strategy="steps",
    logging_steps=10,
    per_device_train_batch_size=8,
    num_train_epochs=10,
    save_total_limit=1,
    fp16=False,
    report_to="wandb",  # WandB에 로그 기록
)

# 트레이너 초기화
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_memory_usage_list = []

    def compute_loss(self, model, inputs, return_outputs=False):
        # Runtime 측정 시작
        start_time = time.time()

        # 메모리 및 GPU 사용량 측정
        process = psutil.Process()
        memory_usage = process.memory_info().rss / (1024 ** 2)  # 메모리 사용량(MB 단위)
        gpu_memory_usage = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
        
        # GPU 메모리 사용량을 리스트에 추가
        if torch.cuda.is_available():
            self.gpu_memory_usage_list.append(gpu_memory_usage)
        
        # 손실 계산 및 outputs 가져오기
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        
        # Runtime 측정 종료
        end_time = time.time()
        runtime = end_time - start_time
        
        # WandB에 로그 기록
        wandb.log({
            "train/loss": loss.item(),
            "train/runtime": runtime,
            "train/memory_usage_MB": memory_usage,
            "train/gpu_memory_usage_MB": gpu_memory_usage,
        })
        
        return (loss, outputs) if return_outputs else loss

    def log_average_gpu_memory_usage(self):
        # GPU 메모리 사용량의 평균을 계산하고 WandB에 로그 기록
        if self.gpu_memory_usage_list:
            avg_gpu_memory_usage = sum(self.gpu_memory_usage_list) / len(self.gpu_memory_usage_list)
            wandb.log({"train/average_gpu_memory_usage_MB": avg_gpu_memory_usage})
            print(f"Average GPU Memory Usage: {avg_gpu_memory_usage:.2f} MB")

# 전체 훈련 시간 측정 시작
overall_start_time = time.time()

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collator,
)

# 학습 시작
train_result = trainer.train()

# 전체 훈련 시간 계산 및 로그 기록
overall_end_time = time.time()
total_training_time = (overall_end_time - overall_start_time) / 60  # 분 단위 변환
wandb.log({"train/total_training_time_min": total_training_time})

# 전체 훈련 시간 콘솔 출력
print(f"Total Training Time: {total_training_time:.2f} minutes")

# 평균 GPU 메모리 사용량 로그 기록
trainer.log_average_gpu_memory_usage()

# 모델 저장
trainer.save_model("./fine_tuned_therapist_chatbot")

# 학습 중 로그 히스토리 확인
df = pd.DataFrame(trainer.state.log_history)
print(df)  # 로그 기록 출력 (손실 값이 기록되었는지 확인)

# WandB 로깅 종료
wandb.finish()
