import json
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from trl import SFTConfig, SFTTrainer
import wandb
from transformers import TrainerCallback

# WandB 초기화
wandb.init(project="therapist-chatbot", name="fine-tuning")

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

# SFT 설정 및 트레이너 정의
sft_config = SFTConfig(
    output_dir="./results",
    eval_strategy="epoch",  # 매 epoch마다 평가
    logging_strategy="steps",  # steps 단위로 로그 남기기
    logging_steps=10,  # 10 스텝마다 로깅
    eval_steps=10,  # 10 스텝마다 평가
    per_device_train_batch_size=8,  # 배치 크기 설정
    per_device_eval_batch_size=8,
    num_train_epochs=10,  # 에폭 수 10으로 설정
    save_total_limit=1,
    fp16=False,  # FP16 비활성화
    run_name="therapist-fine-tuning-run"  # WandB run name 설정
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=sft_config,
    data_collator=collator,
)

# WandB와 통합된 TrainerCallback 정의
class WandbCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            wandb.log(logs)

    def on_epoch_end(self, args, state, control, **kwargs):
        # 에폭 종료 시 손실 값을 WandB에 기록
        if len(state.log_history) > 0:
            for log in state.log_history:
                if "loss" in log:
                    logs = {"train/loss": log["loss"], "train/epoch": state.epoch}
                    wandb.log(logs)

# WandB 콜백 추가
trainer.add_callback(WandbCallback)

# 학습 시작
train_result = trainer.train()

# 평가 데이터셋으로 평가 실행
eval_metrics = trainer.evaluate()

# WandB에 eval 결과 로깅
wandb.log({"eval/loss": eval_metrics.get('eval_loss', 0), "eval/epoch": eval_metrics.get('epoch', 0)})

# 모델 저장
trainer.save_model("./fine_tuned_therapist_chatbot")

# 학습 중 로그 히스토리 확인
df = pd.DataFrame(trainer.state.log_history)
print(df)  # 로그 기록 출력 (손실 값이 기록되었는지 확인)

# 학습 및 평가 결과 로깅
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)

trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", eval_metrics)

# WandB 로깅 종료
wandb.finish()
