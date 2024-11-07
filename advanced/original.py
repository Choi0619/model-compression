import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import wandb
from dotenv import load_dotenv  # .env 파일을 불러오기 위한 라이브러리
import google.generativeai as genai  # Google Gemini API SDK

# .env 파일에서 API 키 불러오기
load_dotenv()  # .env 파일 로드
google_api_key = os.getenv("GOOGLE_API_KEY")  # .env 파일에 있는 GOOGLE_API_KEY 값 가져오기

# Google Generative AI API 설정
genai.configure(api_key=google_api_key)

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

# Google Gemini 모델 정의
model = genai.GenerativeModel('gemini-1.5-flash')  # 또는 'gemini-1.5-pro'

# 전처리 함수 정의
def preprocess_function(examples):
    responses = []
    for example in examples:
        input_text = example['input']
        
        # Google Gemini 모델을 사용해 응답 생성
        response = model.generate_content(input_text)
        
        # 응답 텍스트 추출
        output_text = response.text if hasattr(response, 'text') else "No response"
        responses.append({"input": input_text, "output": output_text})
    
    return responses

# Google Gemini API를 통해 데이터셋 응답 생성
train_dataset = preprocess_function(train_data)
val_dataset = preprocess_function(val_data)

# WandB 로깅 추가 및 평가
wandb.log({"train_data_count": len(train_dataset), "val_data_count": len(val_dataset)})

# 학습된 모델 응답 및 평가 결과 출력
print("Generated Responses from Train Dataset:", train_dataset[:5])
print("Generated Responses from Validation Dataset:", val_dataset[:5])

# WandB 로깅 종료
wandb.finish()
