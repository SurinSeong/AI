import os
import google.generativeai as genai

## api 키 불러오기 ##
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=gemini_api_key)

## 모델 설정 ##
model = genai.GenerativeModel(model_name='gemini-1.5-flash',
                              generation_config={'temperature':0,
                                                 'max_output_tokens':5000})

## Chat session 객체 반환 ##
chat_session = model.start_chat(history=[])

## 질문 받고 답변 확인하기 ##
query = input('무엇이든 물어보세요! ')

response = chat_session.send_message(query)
print(response)
print(f'AI : {response.text}')