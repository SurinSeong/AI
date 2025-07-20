import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# ChatOpenAI 모델 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9, max_tokens=1000)

# DALL-E 이미지 생성을 위한 프롬프트 템플릿 정의
prompt = PromptTemplate.from_template(
    "Generate a detailed IMAGE GENERATION prompt for DALL-E based on the following description. "
    "Return only the prompt, no intro, no explanation, no chatty, no markdown, no code block, no nothing. Just the prompt"
    "Output should be less than 1000 characters. Write in English only."
    "Image Description: \n{image_desc}",
)

# 프롬프트, LLM, 출력 파서를 연결하는 체인 생성
chain = prompt | llm | StrOutputParser()

query = input("어떤 그림을 원하시나요?\n")

# 체인 실행
image_prompt = chain.invoke(
    {"image_desc": query}
)

print(image_prompt)

## 이미지 프롬프트를 DallEAPIWrapper에 입력하기
# DALL-E API 래퍼 가져오기
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from IPython.display import Image

# DALL-E API 래퍼 초기화
# model: 사용할 DALL-E 모델 버전
# size: 생성할 이미지 크기
# quality: 이미지 품질
# n: 생성할 이미지 수
dalle = DallEAPIWrapper(model="dall-e-3", size="1024x1024", quality="standard", n=1)

# dalle.run()을 사용해서 실제 이미지 생성
image_url = dalle.run(image_prompt)

# 생성된 이미지 표시하기
print(image_url)
Image(url=image_url, width=500)