import os
import streamlit as st
import google.generativeai as genai
from PIL import Image

from prac_utils import load_model, making_id, reset_session_state, get_session_history

## api 키 불러오기 ##
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=gemini_api_key)

model = load_model()

## 세션별 이력 관리하기 ##
if 'store' not in st.session_state:
    st.session_state.store = dict()

# 히스토리 관리
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = model.start_chat(history=[])

# 아이디 관리
if 'session_id' not in st.session_state:
    st.session_state.session_id = ""
    
# 로그인 유무 관리
if 'is_logged_in' not in st.session_state:
    st.session_state.is_logged_in = False

## 사이드바 꾸미기 ## ------------------------------------------------
# 사이드바에 아이디 넣는 곳 만들기
with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        if st.button('닉네임 생성'):
            making_id()
        
    with col2:
        if st.button('로그아웃'):
            reset_session_state()
            st.rerun()
            
    session_id = st.text_input('Session ID', value=st.session_state['session_id'], key='session_id')

    # 로그인 버튼을 누르면
    if st.button('로그인'):
        if st.session_state.session_id != "":
            st.session_state.is_logged_in = True # 로그인 상태 저장하기
            get_session_history(session_id, st.session_state.chat_history)
        else:
            st.markdown("Session ID를 입력하세요.")
        
        # 로그인 성공 후 세션 상태 확인하기
        if st.session_state.get('is_logged_in', False):
            st.markdown(f"현재 로그인된 닉네임: {st.session_state.session_id}")
        else:
            st.markdown('로그인을 해주세요.')
            
    ## 도시 선택하기 ##
    st.title("🧐 옵션을 선택하면 빠르게 추천해드려요!\n")
    st.subheader("지역을 선택하세요! 해당 지역의 맛집을 찾아드리겠습니다.\n")
    
    # 체크박스 사용하기
    local_jeju_city = st.checkbox('제주시')
    local_seogwipo_city = st.checkbox('서귀포시')
    st.write('\n')
    
    # PNG 이미지 삽입하기
    img= Image.open(r'C:\Users\tjdtn\inflearn-llm-application\big-contest-streamlit\제주도 지도.png')
    st.image(img, caption='제주도 지도', use_column_width=True)

# -------------------------------------------------------------------------
## 사용자 입력에 따른 검색 답변 보여주기 ##
if ('is_logged_in' not in st.session_state) or (not st.session_state.is_logged_in):
    # 닉네임 생성해달라고 해주기
    with st.chat_message('ai'):
        st.markdown('닉네임을 생성하고 로그인해주세요.')

else:
    # 첫 방문일 때 메시지 출력하기
    if len(st.session_state.store[session_id].history) == 0:
        if prompt := st.chat_input('반갑습니다. 어떤 음식점을 찾고 계신가요?'):
            with st.spinner('음식점을 찾는 중입니다...'):
                
                # 제주 선택했는데 서귀포 물어보면
                if (local_jeju_city) and (not local_seogwipo_city) and ('서귀포' in prompt):
                    response = '제주시에 있는 음식점만 추천해드릴 수 있어요.\n서귀포시에 있는 음식점을 추천받고 싶다면 서귀포시에 체크해주세요.'
                    with st.chat_message('ai'):
                        st.markdown(response)
                
                # 서귀포 선택했는데 제주 물어보면    
                elif (local_seogwipo_city) and (not local_jeju_city) and ('제주' in prompt):
                    response = "서귀포시에 있는 음식점만 추천해드릴 수 있어요.\n제주시에 있는 음식점을 추천받고 싶다면 제주시에 체크해주세요."
                    with st.chat_message('ai'):
                        st.markdown(response)
                    
                # 잘 선택했다면
                else:
                    response = st.session_state.chat_history.send_message(prompt)
                    # 대화이력 출력하기
                    for content in st.session_state.chat_history.history:
                        with st.chat_message('ai' if content.role == 'model' else 'user'):
                            st.markdown(content.parts[0].text)
    else:
        if prompt := st.chat_input('다시 오셨네요! 더 궁금한 것이 있으신가요?'):        
            with st.spinner('음식점을 찾는 중입니다...'):
                
                # 제주 선택했는데 서귀포 물어보면
                if (local_jeju_city) and (not local_seogwipo_city) and ('서귀포' in prompt):
                    response = '제주시에 있는 음식점만 추천해드릴 수 있어요.\n서귀포시에 있는 음식점을 추천받고 싶다면 서귀포시에 체크해주세요.'
                    with st.chat_message('ai'):
                        st.markdown(response)
                
                # 서귀포 선택했는데 제주 물어보면    
                elif (local_seogwipo_city) and (not local_jeju_city) and ('제주' in prompt):
                    response = "서귀포시에 있는 음식점만 추천해드릴 수 있어요.\n제주시에 있는 음식점을 추천받고 싶다면 제주시에 체크해주세요."
                    with st.chat_message('ai'):
                        st.markdown(response)
                     
                # 잘 선택했다면        
                else:
                    response = st.session_state.store[session_id].send_message(prompt)
                    # 대화 이력 출력하기
                    for content in st.session_state.store[session_id].history:
                        with st.chat_message('ai' if content.role == 'model' else 'user'):
                            st.markdown(content.parts[0].text)



print(f'ID 내역\n>> {st.session_state.session_id}, {st.session_state.is_logged_in}')    
print(f'채팅 내역을 보여드립니다.\n>> {st.session_state.store}')
print(f'\n각 답변을 나눠서 보여드립니다.\n>> {st.session_state.chat_history}')