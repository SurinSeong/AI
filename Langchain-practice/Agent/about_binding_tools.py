import os
from dotenv import load_dotenv

load_dotenv()

import re
import requests
from bs4 import BeautifulSoup
from langchain.agents import tool

# 도구 정의하기
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)


@tool
def add_function(a: float, b: float) -> float:
    """Adds two numbers together."""
    return a + b


@tool
def naver_news_crawl(news_url: str) -> str:
    """Crawls a Naver (naver.com) news article and returns the body content."""
    # HTTP GET 요청 보내기
    response = requests.get(news_url)

    # 요청이 성공했는지 확인하기
    if response.status_code == 200:
        # BeautifulSoup를 사용해서 HTML 파싱하기
        soup = BeautifulSoup(response.text, "html.parser")

        # 원하는 정보 추출
        title = soup.find("h2", id="title_area").get_text()
        content = soup.find("div", id="contents").get_text()
        cleaned_title = re.sub(r"\n{2,}", "\n", title)
        cleaned_content = re.sub(r"\n{2,}", "\n", content)
    else:
        print(f"HTTP 요청 실패. 응답 코드: {response.status_code}")

    return f"{cleaned_title}\n{cleaned_content}"


tools = [get_word_length, add_function, naver_news_crawl]

## bind_tools()로 LLM에 도구 바인딩
# llm 모델에 bind_tools()를 사용해서 도구 바인딩

from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 모델 생성
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 도구 바인딩
llm_with_tools = llm.bind_tools(tools)

query = input("무엇이 궁금하신가요?\n")

# 실행 결과
print(llm_with_tools.invoke(query).tool_calls)

# llm_with_tools와 JsonOutputToolsParser를 연결해서 tool_calls를 parsing해서 결과를 확인한다.
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser

# 도구 바인딩 + 도구 파서
chain = llm_with_tools | JsonOutputToolsParser(tools=tools)

# 실행 결과
tool_call_results = chain.invoke(query)

print(tool_call_results, end="\n\n==========\n\n")

# 첫 번째 도구 호출 결과
single_result = tool_call_results[0]

# 도구 이름
print(single_result["type"])

# 도구 인자
print(single_result["args"])

# execute_tool_calls : 도구를 찾아 args를 전달하여 도구를 실행
    # type : 도구의 이름, args : 도구에 전달되는 인자

def execute_tool_calls(tool_call_results):
    """
    도구 호출 결과를 실행하는 함수

    :param tool_call_results: 도구 호출 결과 리스트
    :param tools: 사용 가능한 도구 리스트
    """
    # 도구 호출 결과 리스트를 순회
    for tool_call_result in tool_call_results:
        # 도구의 이름과 인자를 추출
        tool_name = tool_call_result["type"]
        tool_args = tool_call_result["args"]

        # 도구 이름과 일치하는 도구를 찾아 실행한다.
        # next() 함수를 사용해서 일치하는 첫 번째 도구를 찾는다.
        matching_tool = next((tool for tool in tools if tool.name == tool_name), None)

        if matching_tool:
            # 일치하는 도구를 찾았다면 해당 도구를 실행
            result = matching_tool.invoke(tool_args)
            # 실행 결과를 출력
            print(f"[실행도구] {tool_name}\n[실행결과] {result}")

        else:
            # 일치하는 도구를 찾지 못했다면 경고 메시지를 출력한다.
            print(f"경고: {tool_name}에 해당하는 도구를 찾을 수 없음.")

# 도구 호출 실행
# 이전에 얻은 tool_call_results를 인자로 전달해서 함수를 실행한다.
execute_tool_calls(tool_call_results)

# --------------------------------------------------------------
# bind_tools + Parser + Execution
# 1. 모델에 도구를 바인딩
# 2. 도구 호출 결과를 파싱하는 파서
# 3. 도구 호출 결과를 실행하는 함수
chain = llm_with_tools | JsonOutputToolsParser(tools=tools) | execute_tool_calls

chain.invoke(query)
chain.invoke("114.5 + 121.2")
chain.invoke("뉴스 기사 내용을 크롤링해줘: https://n.news.naver.com/mnews/hotissue/article/092/0002347672?type=series&cid=2000065")

# --------------------------------------------------------------
# bind_tools를 Agent & AgentExecutor로 대체하기
# AgentExecutor : 실제로 llm 호출, 올바른 도구로 라우팅, 실행, 모델 재호출 등을 위한 실행 루프를 생성
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Agent 프롬프트 생성하기
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but don't know current events",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# 모델 생성
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

from langchain.agents import create_tool_calling_agent, AgentExecutor

# 이전에 정의한 도구를 사용해서 agent를 생성한다.
agent = create_tool_calling_agent(llm, tools, prompt)

# AgentExecutor 생성
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

# Agent 실행하기
result = agent_executor.invoke({"input": query})

# 결과 확인하기
print(result["output"])

result = agent_executor.invoke(
    {
        "input": "뉴스 기사를 요약해 줘: https://n.news.naver.com/mnews/hotissue/article/092/0002347672?type=series&cid=2000065"
    }
)
print(result["output"])
