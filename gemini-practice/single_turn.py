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

## 질문 받아서 답변 추출하기 ##
query = input('무엇이든 물어보세요!\n')
response = model.generate_content(query)
print(response.text)