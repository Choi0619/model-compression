import json
import pandas as pd
from sklearn.model_selection import train_test_split
import wandb
import openai
from dotenv import load_dotenv

# 환경 변수 로드 (API 키를 .env 파일에서 불러옴)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

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

# 모델 호출 함수 정의
def generate_response(input_text, model="gpt-4.0-mini"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "user", "content": input_text}
        ]
    )
    return response.choices[0].message['content']

# WandB에 학습 손실 로깅을 위한 함수
def log_training_progress(train_data, model="gpt-4.0-mini"):
    total_loss = 0
    for i, pair in enumerate(train_data):
        input_text = pair["input"]
        expected_output = pair["output"]

        # 모델로부터 응답 생성
        model_response = generate_response(input_text, model=model)

        # 예측과 실제의 차이 계산 (여기서는 간단히 문자열 유사도를 평가 지표로 사용)
        loss = len(set(expected_output) ^ set(model_response)) / len(set(expected_output + model_response))
        total_loss += loss

        # 10번째 데이터마다 WandB에 손실 로그 기록
        if (i + 1) % 10 == 0:
            avg_loss = total_loss / (i + 1)
            wandb.log({"train/loss": avg_loss})
            print(f"Step {i + 1}, Loss: {avg_loss}")

    avg_total_loss = total_loss / len(train_data)
    print(f"Total Average Loss: {avg_total_loss}")
    wandb.log({"train/total_avg_loss": avg_total_loss})

# 평가 함수 정의
def evaluate(val_data, model="gpt-4.0-mini"):
    total_loss = 0
    for i, pair in enumerate(val_data):
        input_text = pair["input"]
        expected_output = pair["output"]

        # 모델로부터 응답 생성
        model_response = generate_response(input_text, model=model)

        # 손실 계산
        loss = len(set(expected_output) ^ set(model_response)) / len(set(expected_output + model_response))
        total_loss += loss

    avg_val_loss = total_loss / len(val_data)
    print(f"Validation Loss: {avg_val_loss}")
    wandb.log({"eval/loss": avg_val_loss})

# 학습 시작
log_training_progress(train_data, model="gpt-4.0-mini")

# 평가 시작
evaluate(val_data, model="gpt-4.0-mini")

# WandB 로깅 종료
wandb.finish()
