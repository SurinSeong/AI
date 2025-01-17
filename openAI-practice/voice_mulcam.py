import os
import streamlit as st
from openai import OpenAI
import openai

client = OpenAI(
    api_key = ''
)

st.title("OpenAI's Text to Audio Response.")

# 이미지 가져오기
st.image('https://wikidocs.net/images/page/215361/%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5%EC%84%B1%EC%9A%B0.jpg', width=200)

## 인공지능 성우 선택박스 생성
# 인공지능 성우 모음
options = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
# 드롭다운 선택 박스 만들기
selected_option = st.selectbox('성우를 선택해 주세요 :', options)

## 인공지능 성우에게 명령 프롬프트 전달하기
# 가이드 설정
default_text = '오늘은 생활의 꿀팁을 알아보겠습니다.'
user_prompt = st.text_area('인공지능 성우가 읽을 스크립트를 입력해주세요.',
                            value=default_text, height=200)

## 버튼 만들기
## Generate Audio button >> 버튼 누르면 True (1) --> if문 실행
if st.button('Generate Audio'):
    # TEXT >> VOICE 생성
    audio_response = \
    client.audio.speech.create(
        model='tts-1',
        voice=selected_option,
        input = user_prompt
    )
    
    # VOICE(음성) >> mp3 파일 저장
    audio_content = audio_response.content
    with open('temp_mulcam_audio.mp3', 'wb') as audio_file:
        audio_file.write(audio_content)
        
    # mp3 파일 재생
    st.audio('temp_mulcam_audio.mp3', format='audio/mp3')
