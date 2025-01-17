import random
import streamlit as st

## 닉네임 생성하기 ##
def generate_random_id():
    # 닉네임 (id) 생성
    jeju_nicknames = [
        "한라산바람", "오름여행자", "바다향기", "감귤연인", "돌하르방친구", "푸른섬나그네", 
        "섭지코지연인", "해녀이야기", "제주바다빛", "감성제주러", "한치도사", "제주하늘", 
        "돌담길여행자", "조랑말의꿈", "바람의섬", "우도탐험가", "평화의바다", "제주푸름", 
        "오름의숨결", "비양도의꿈", "올레길여행자", "새별오름러버", "제주향기", "애월바다러버", 
        "성산일출연인", "한라봉나그네", "비자림의추억", "해안도로러버", "구좌바다바람", "용눈이오름"
    ]
    return random.choice(jeju_nicknames)
    
def making_id():
    created_id = generate_random_id()
    st.session_state['session_id'] = created_id
 
## 모델 가져오기 ##
# 데이터 캐싱 이용
import google.generativeai as genai

@st.cache_resource
def load_model():
    model = genai.GenerativeModel('gemini-1.5-flash',
                                  generation_config={"temperature": 0,
                                                     "max_output_tokens": 5000})
    print('model loaded...')
    return model

model = load_model()
   
## 초기화 함수 ##
def reset_session_state():
    # st.session_state['messages'] = []
    st.session_state['session_id'] = ""  # ID 초기화
    st.session_state['is_logged_in'] = False
    st.session_state['chat_history'] = model.start_chat(history=[])
    
## 세션 기록 불러오기 ##
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_community.chat_message_histories import ChatMessageHistory

def get_session_history(session_id, chat_history):
    # 세션 아이디가 전체 내역 안에 없다면
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = chat_history
    return st.session_state.store[session_id]
    
# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in st.session_state.store:  # 세션 ID가 store에 없는 경우
#         # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
#         st.session_state.store[session_id] = ChatMessageHistory()
#     return st.session_state.store[session_id]  # 해당 세션 ID에 대한 세션 기록 반환
 