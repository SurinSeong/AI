from openai import OpenAI
import streamlit as st
import requests

# open api key 값 세팅
cilent = OpenAI(
    api_key=''
)

# 주어진 이미지 주소로부터 GPT4V의 설명을 얻는 함수
def describe(image_url):
    response = cilent.chat.completions.create(
        model='gpt-4-vision-preview',
        messages=[
            {'role':'user',
             'content':[
                 {'type':'text',
                  'text':'이 이미지에 대해서 알려줘'},
                 {'type':'image_url',
                  'image_url':{
                     'url':image_url,
                 },
                 },
             ],
             },
        ],
        max_tokens=1024,
    )
    return response.choices[0].message.content

# 웹사이트 상단에 노출된 웹사이트 제목
st.title('AI 도슨트 : 이미지를 설명해드립니다.')

# st.text_area()는 사용자 입력을 받는 커다란 텍스트 칸을 만들어 줌.
# height : 이 텍스트 칸의 높이
input_url = st.text_area('여기에 이미지 주소를 입력하세요.', height=30)

# st.button()을 클릭하면 >> st.button() 값이 True로 변경
# if 문이 실행된다.
if st.button('해설'):
    # st.text_area()의 값이 존재하면 input_url 값이 True로 되고 if문 실행
    if input_url:
        try:
            # st.image() : 기본적으로 이미지 주소로부터 이미지를 웹사이트 화면에 생성한다.
            st.image(input_url, width=300)
            # describe() 호출 --> GPT4V 출력결과 반환
            result = describe(input_url)
            
            # st.success() : 텍스트를 웹사이트 화면에 출력 >> 초록색 바탕
            st.success(result)
            
        except:
            st.error('요청 오류가 발생했습니다.')
    else:
        # 화면 상으로 노란색 배경을 출력함.
        st.warning('텍스트를 입력하세요.')
        