import os
from dotenv import load_dotenv

load_dotenv()

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

from langchain_community.tools.tavily_search import TavilySearchResults

# 도구 생성하기
tavily_tool = TavilySearchResults(
    max_results=6,
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    search_depth="basic",
    include_domains=["github.io", "wikidocs.net"],
)

# 도구 실행
query = input("어떤 것이 궁금하신가요?\n")

print(tavily_tool.invoke({"query": query}))