import os
import google.generativeai as genai

## api 키 불러오기 ##
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=gemini_api_key)

## 시스템 인스트럭션 ##
system_instruction = "당신은 유치원 선생님입니다. 사용자는 유치원생입니다. 쉽고 친절하게 이야기하되, 3문장 이내로 짧게 이야기해주세요."

## 모델 설정 ##
model = genai.GenerativeModel(model_name='gemini-1.5-flash',
                              generation_config={'temperature':0,
                                                 'max_output_tokens':5000},
                              system_instruction=system_instruction)

## Chat session 객체 반환 ##
chat_session = model.start_chat(history=[])

## 질문 받고 답변 확인하기 ##
query = input('무엇이든 물어보세요! ')

response = chat_session.send_message(query)
print(f'USER : {query}')
print(f'AI : {response.text}')