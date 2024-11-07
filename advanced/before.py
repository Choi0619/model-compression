import os
import streamlit as st
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from transformers import pipeline
import time

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 모델과 감정 분석 설정
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# 대화 맥락 유지를 위한 메모리 설정 (LangChain 메모리 활용)
memory = ConversationBufferMemory(return_messages=True)  # 대화 기록을 유지

# corpus.json 파일 불러오기
with open('corpus.json', 'r', encoding='utf-8') as file:
    corpus = json.load(file)

# 입력-출력 쌍 준비
data_pairs = []
for i in range(0, len(corpus) - 1, 2):  # user와 therapist 쌍으로 진행
    if corpus[i]['role'] == 'user' and corpus[i + 1]['role'] == 'therapist':
        input_text = corpus[i]['content']  # 사용자 입력
        output_text = corpus[i + 1]['content']  # 치료사 응답
        data_pairs.append({"input": input_text, "output": output_text})

# 학습 데이터 출력 (데모 용도)
for pair in data_pairs:
    memory.save_context({"input": pair['input']}, {"output": pair['output']})

# Streamlit UI 설정 - 페이지 타이틀
st.set_page_config(page_title="마음 쉼터 상담 챗봇", page_icon="🌸")

# 스타일링 CSS 적용
st.markdown("""
    <style>
    body { background-color: #FAF3F3; }
    .chat-container {
        max-width: 700px;
        margin: auto;
        padding: 5px;
        border: 1px solid #e6e6e6;
        border-radius: 15px;
        background-color: #FFFDFD;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    .chat-row {
        display: flex;
        margin: 5px;
        width: 100%;
    }
    .row-reverse {
        flex-direction: row-reverse;
    }
    .chat-bubble {
        font-family: "Source Sans Pro", sans-serif;
        border: 1px solid transparent;
        padding: 10px 15px;
        margin: 0px 7px;
        max-width: 70%;
        font-size: 15px;
    }
    .ai-bubble {
        background: rgb(240, 242, 246);
        border-radius: 10px;
    }
    .human-bubble {
        background: linear-gradient(135deg, rgb(0, 178, 255) 0%, rgb(0, 106, 255) 100%);
        color: white;
        border-radius: 20px;
    }
    .input-area {
        margin-top: 20px;
    }
    .feedback-container {
        max-width: 700px;
        margin: auto;
        padding: 5px;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        background-color: #FFFCF9;
    }
    .submit-button {
        background-color: #F28A8A;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 16px;
        cursor: pointer;
        margin-top: 10px;
    }
    .submit-button:hover {
        background-color: #FF6B6B;
    }
    </style>
""", unsafe_allow_html=True)

# UI 시작 - 챗봇 타이틀 및 설명
st.title("🌸 마음 쉼터 상담 챗봇 🌸")
st.write("안녕하세요! 따뜻한 마음으로 귀 기울여 드릴게요. 언제든지 마음을 나눠보세요.")

# 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False
if "show_thank_you" not in st.session_state:
    st.session_state.show_thank_you = False

# 채팅 기록 표시
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for message in st.session_state.messages:
    role, content = message["role"], message["content"]
    if role == "user":
        st.markdown(f"<div class='chat-row row-reverse'><div class='chat-bubble human-bubble'>{content}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-row'><div class='chat-bubble ai-bubble'>{content}</div></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# 사용자 입력 처리
if prompt := st.chat_input("저에게 본인의 마음을 털어놓아보세요..."):
    # 감정 분석 수행
    sentiment_result = sentiment_analyzer(prompt)[0]
    sentiment = sentiment_result['label']

    # 감정 분석 결과에 따라 톤 설정
    if sentiment in ["1 star", "2 stars"]:  # 부정적인 감정
        tone = "위로와 공감을 드리는 존댓말로"
    elif sentiment in ["4 stars", "5 stars"]:  # 긍정적인 감정
        tone = "따뜻하고 격려하는 존댓말로"
    else:  # 중립적인 감정
        tone = "편안하고 공감적인 존댓말로"

    # 사용자 입력 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"<div class='chat-row row-reverse'><div class='chat-bubble human-bubble'>{prompt}</div></div>", unsafe_allow_html=True)

    # LangChain 메모리를 사용해 대화 히스토리를 포함한 프롬프트 생성
    conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in memory.chat_memory.messages])
    question_template = PromptTemplate(
        input_variables=["tone", "conversation_history", "user_input"],
        template="{tone} 말투로, 넘버링 없이 마치 가까운 사람과 이야기하듯 편하게 조언해 주세요. 예를 들어, '저도 가벼운 산책이나 운동을 할 때 기분이 많이 나아지더라고요.' 같은 방식으로 답변을 작성해 주세요. 이전 대화를 참고하여 답변해 주세요. 이전 대화: {conversation_history} 사용자 질문: {user_input}"
    )

    # 템플릿에 값 적용
    formatted_prompt = question_template.format(
        tone=tone, conversation_history=conversation_history, user_input=prompt
    )

    # GPT-4 모델을 사용하여 응답 생성
    answer = llm([HumanMessage(content=formatted_prompt)]).content
    st.markdown(f"<div class='chat-row'><div class='chat-bubble ai-bubble'>{answer}</div></div>", unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # LangChain 메모리에 대화 저장
    memory.save_context({"input": prompt}, {"output": answer})

# 상담 종료 버튼 및 피드백 창
if st.button("상담 종료"):
    st.session_state.feedback_submitted = False
    st.session_state.show_thank_you = False  # 피드백 후 감사 메시지 초기화
    st.markdown("<div class='feedback-container'>", unsafe_allow_html=True)
    st.subheader("상담이 도움이 되셨나요?")
    feedback = st.radio("상담 경험을 평가해주세요:", ("", "매우 만족", "만족", "보통", "불만족", "매우 불만족"), index=0)
    
    # 제출 버튼 추가
    if feedback and st.button("제출", key="submit_feedback"):
        st.session_state.feedback_submitted = True
        st.session_state.show_thank_you = True

# 피드백 제출 후 감사 메시지 표시
if st.session_state.show_thank_you:
    st.success("피드백을 주셔서 감사합니다! 상담 챗봇의 개선에 도움이 됩니다.")
    time.sleep(2)  # 2초 동안 감사 메시지 표시
    st.session_state.show_thank_you = False  # 메시지 초기화
    st.experimental_rerun()  # 페이지 새로고침으로 피드백 창 닫기