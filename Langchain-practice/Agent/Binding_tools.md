# Binding Tools 도구 바인딩

## LLM에 도구 바인딩하기

- LLM 모델이 도구를 호출할 수 있기 위해서는 chat 요청을 할 때 모델에 도구 스키마를 전달해야 한다.
- 도구 호출 (tool calling) 기능을 지원하는 langchain chat model은 `bind_tools()` 메서드를 구현해서 LangChain 도구 객체, Pydantic 클래스 또는 JSON 스키마 목록을 수신하고 공급자별 예상 형식으로 채팅 모델에 바인딩한다.
- 바인딩된 Chat Model의 후속 호출은 모델 API에 대한 모든 호출에 도구 스키마를 포함한다.

## LLM에 바인딩할 Tool 정의

- `get_word_length` : 단어의 길이를 반환하는 함수
- `add_function` : 두 숫자를 더하는 함수
- `naver_news_crawl` : 네이버 뉴스 기사를 크롤링하여 본문의 내용을 반환


