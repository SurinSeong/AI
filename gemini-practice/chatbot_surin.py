import os
import streamlit as st
from PIL import Image
import google.generativeai as genai

from utils_surin import making_id, reset_session_state, get_session_history, main

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import ChatMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

## api 키 불러오기 ##
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv('GOOGLE_API_KEY') ### 설정하신 걸로 바꿔주시면 됩니다. ###
genai.configure(api_key=gemini_api_key)

## chatbot UI 설정
st.set_page_config(page_title='제주도 맛집', page_icon="🏆",initial_sidebar_state="expanded")
st.title('제주도 음식점 탐방!')
st.subheader("누구와 제주도에 오셨나요? 제주도 맛집 추천해드려요~")

st.write("")

st.write("#연인#아이#친구#부모님#혼자#반려동물 #데이트#나들이#여행#일상#회식#기념일...")
st.write("")

## Embedding Model 로드하기 ##
embedding_function = HuggingFaceEmbeddings(model_name='jhgan/ko-sroberta-multitask')

## ChromaDB 불러오기 ##
recommendation_store = Chroma(
    collection_name='jeju_store_mct_keyword_6',
    embedding_function=embedding_function,
    persist_directory= r'C:\Users\tjdtn\inflearn-llm-application\big-contest-streamlit\chromadb\mct_keyword_v6'
)
# metadata 설정
metadata = recommendation_store.get(include=['metadatas'])

# ------------------------------------------------------------
## 세션별 이력 관리하기 ##
if 'store' not in st.session_state:
    st.session_state.store = dict()

# 히스토리 관리
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# 아이디 관리
if 'session_id' not in st.session_state:
    st.session_state.session_id = ""
    
# 로그인 유무 관리
if 'is_logged_in' not in st.session_state:
    st.session_state.is_logged_in = False
    
## 사이드바 꾸미기 ------------------------------------------------------
# 사이드바에 아이디 넣는 곳을 만들기
with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        if st.button('닉네임 생성'):
            making_id()
            
    with col2:
        if st.button('로그아웃'):
            reset_session_state()
            st.rerun()
            
    session_id = st.text_input('Session ID', value=st.session_state.session_id, key='session_id')
    
    # 로그인 버튼을 누르면
    if st.button('로그인'):
        # session id가 작성되었다면
        if st.session_state.session_id != "":
            st.session_state.is_logged_in = True # 로그인 상태를 True
            get_session_history(session_id)
            
        # session id가 작성되지 않았다면
        else:
            st.markdown("Session ID를 입력하세요.")
            
        # 로그인 성공 후 세션 상태 확인
        if st.session_state.get('is_logged_in', False):
            st.markdown(f'현재 로그인된 닉네임: {st.session_state.session_id}')
        else:
            st.markdown('로그인을 해주세요.')
            
    ## 도시 체크박스
    st.title("옵션을 선택하면 빠르게 추천해드려요!\n")
    st.subheader("지역을 선택하세요! 해당 지역의 맛집을 찾아드리겠습니다.\n")
    
    # 체크박스 사용
    local_jeju_city = st.checkbox('제주시')
    local_seogwipo_city = st.checkbox('서귀포시')
    st.write('\n')
    
    # PNG 이미지 삽입하기
    img= Image.open(r'C:\Users\tjdtn\inflearn-llm-application\big-contest-streamlit\제주도 지도.png')
    st.image(img, caption='제주도 지도', use_column_width=True)
    
# ------------------------------------------------------------------------
## 사용자의 입력에 따른 검색 답변 보여주기 ##
# 로그인 안되었다면
if ('is_logged_in' not in st.session_state) or (not st.session_state.is_logged_in):
    # 닉네임 생성해달라고 해주기
    with st.chat_message('ai'):
        st.markdown('닉네임을 생성하고 로그인해주세요.')
# 로그인 되었다면
else:
    # 새로운 닉네임인 경우
    if len(st.session_state.store[session_id].messages) == 0:
        if prompt := st.chat_input('반갑습니다. 어떤 음식점을 찾고 계신가요?'):
            # 1. 질문 저장
            st.session_state.chat_history.append(ChatMessage(role='user', content=prompt))
                
            # 2. 답변 기다리기
            with st.spinner('음식점을 찾는 중입니다...'):
                # 1) 제주 선택 --> 서귀포 언급
                if (local_jeju_city) and (not local_seogwipo_city) and ('서귀포' in prompt):
                    response = '제주시에 있는 음식점만 추천해드릴 수 있어요.\n서귀포시에 있는 음식점을 추천받고 싶다면 서귀포시에 체크해주세요.'
                    # 3. 답변 저장
                    st.session_state.chat_history.append(ChatMessage(role='assistant', content=response))
                    # 4. 답변 작성
                    with st.chat_message('ai'):
                        st.markdown(response)
                        
                # 2) 서귀포 선택 --> 제주 언급
                elif (local_seogwipo_city) and (not local_jeju_city) and ('제주' in prompt):
                    response = "서귀포시에 있는 음식점만 추천해드릴 수 있어요.\n제주시에 있는 음식점을 추천받고 싶다면 제주시에 체크해주세요."
                    # 3. 답변 저장
                    st.session_state.chat_history.append(ChatMessage(role='assistant', content=response))
                    # 4. 답변 작성
                    with st.chat_message('ai'):
                        st.markdown(response)
                        
                # 3) 제대로된 질문
                else:
                    # 3. 검색형 또는 추천형 또는 기타 질문에 대한 답변 받기
                    main_runnable = RunnableLambda(
                        lambda inputs: main(inputs['question'], inputs['session_id'])
                    )
                    
                    # 실제 Runnable
                    with_message_history = RunnableWithMessageHistory(
                        main_runnable,
                        get_session_history,
                        input_messages_key='question',
                        history_messages_key='history'
                    )
                    
                    # 질문이 들어오면 실행하기
                    response = with_message_history.invoke(
                        {'question':prompt, 'session_id':st.session_state['session_id']},
                        config={'configurable': {'session_id':st.session_state['session_id']}}
                    )
                    
                    # 최종으로 invoke한 내용을 response에 넣고 contents에 저장
                    st.session_state.chat_history.append(ChatMessage(role='assistant', content=response))
                    
                    # 대화 이력 출력하기
                    for message in st.session_state.chat_history:
                        with st.chat_message('ai' if message.role == 'assistant' else 'user'):
                            st.markdown(message.content)
                            
    else:
        if prompt := st.chat_input('또 어떤 음식점을 추천받고 싶으세요?'):
            # 1. 질문 저장
            st.session_state.chat_history.append(ChatMessage(role='user', content=prompt))
                
            # 2. 답변 기다리기
            with st.spinner('음식점을 찾는 중입니다...'):
                # 1) 제주 선택 --> 서귀포 언급
                if (local_jeju_city) and (not local_seogwipo_city) and ('서귀포' in prompt):
                    response = '제주시에 있는 음식점만 추천해드릴 수 있어요.\n서귀포시에 있는 음식점을 추천받고 싶다면 서귀포시에 체크해주세요.'
                    # 3. 답변 저장
                    st.session_state.chat_history.append(ChatMessage(role='assistant', content=response))
                    # 4. 답변 작성
                    with st.chat_message('ai'):
                        st.markdown(response)
                        
                # 2) 서귀포 선택 --> 제주 언급
                elif (local_seogwipo_city) and (not local_jeju_city) and ('제주' in prompt):
                    response = "서귀포시에 있는 음식점만 추천해드릴 수 있어요.\n제주시에 있는 음식점을 추천받고 싶다면 제주시에 체크해주세요."
                    # 3. 답변 저장
                    st.session_state.chat_history.append(ChatMessage(role='assistant', content=response))
                    # 4. 답변 작성
                    with st.chat_message('ai'):
                        st.markdown(response)
                        
                # 3) 제대로된 질문
                else:
                    # 3. 검색형 또는 추천형 또는 기타 질문에 대한 답변 받기
                    main_runnable = RunnableLambda(
                        lambda inputs: main(inputs['question'], inputs['session_id'])
                    )
                    
                    # 실제 Runnable
                    with_message_history = RunnableWithMessageHistory(
                        main_runnable,
                        get_session_history,
                        input_messages_key='question',
                        history_messages_key='history'
                    )
                    
                    # 질문이 들어오면 실행하기
                    response = with_message_history.invoke(
                        {'question':prompt, 'session_id':st.session_state['session_id']},
                        config={'configurable': {'session_id':st.session_state['session_id']}}
                    )
                    
                    # 최종으로 invoke한 내용을 response에 넣고 contents에 저장
                    st.session_state.chat_history.append(ChatMessage(role='assistant', content=response))
                    
                    # 대화 이력 출력하기
                    for message in st.session_state.chat_history:
                        with st.chat_message('ai' if message.role == 'assistant' else 'user'):
                            st.markdown(message.content)
                
                
## 아래부분은 지워도 됩니다 ##
print(f'ID 내역\n>> {st.session_state.session_id}, {st.session_state.is_logged_in}')    
print(f'채팅 내역을 보여드립니다.\n>> {st.session_state.store}')
print(f'\n각 답변을 나눠서 보여드립니다.\n>> {st.session_state.chat_history}')