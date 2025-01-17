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

## 채팅 기록 저장 공간 생성 ##
history = []

user_query = {'role':'user', 'parts':[]}
query = input('무엇이든 물어보세요. ')
user_query['parts'].append(query)

history.append(user_query)
print(f'\n{user_query["role"]} : {user_query["parts"][0]}')
response = model.generate_content(history)
print(f'AI : {response.text}')
history.append(response.candidates[0].content)

print('\n')
print(history)
