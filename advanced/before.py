import json
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
import wandb
import time
import psutil
import torch

# WandB 초기화
wandb.init(project="therapist-chatbot", name="original-training")

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

# 학습 및 검증 세트로 분할 (80-20 비율)
train_data, val_data = train_test_split(data_pairs, test_size=0.2, random_state=42)

# Hugging Face 데이터셋으로 변환
train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
val_dataset = Dataset.from_pandas(pd.DataFrame(val_data))

# 모델과 토크나이저 로드
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

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
val_dataset = val_dataset.map(preprocess_function, batched=True)

# DataCollatorWithPadding 사용
collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    eval_steps=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    save_total_limit=1,
    fp16=False,
    report_to="wandb",  # WandB에 로그 기록
)

# 트레이너 초기화
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Runtime 측정 시작
        start_time = time.time()

        # 메모리 및 GPU 사용량 측정
        process = psutil.Process()
        memory_usage = process.memory_info().rss / (1024 ** 2)  # 메모리 사용량(MB 단위)
        gpu_memory_usage = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
        
        # 손실 계산
        loss = super().compute_loss(model, inputs, return_outputs)
        
        # Runtime 측정 종료
        end_time = time.time()
        runtime = end_time - start_time
        
        # WandB에 로그 기록
        wandb.log({
            "train/runtime": runtime,
            "train/memory_usage_MB": memory_usage,
            "train/gpu_memory_usage_MB": gpu_memory_usage,
        })
        
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
)

# 학습 시작
train_result = trainer.train()

# 평가 데이터셋으로 평가 실행
eval_metrics = trainer.evaluate()

# WandB에 평가 결과 로깅
wandb.log({"eval/loss": eval_metrics.get('eval_loss', 0), "eval/epoch": eval_metrics.get('epoch', 0)})

# 모델 저장
trainer.save_model("./fine_tuned_therapist_chatbot")

# 학습 중 로그 히스토리 확인
df = pd.DataFrame(trainer.state.log_history)
print(df)  # 로그 기록 출력 (손실 값이 기록되었는지 확인)

# WandB 로깅 종료
wandb.finish()
