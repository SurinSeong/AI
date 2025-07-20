# Tools 도구

- 에이전트, 체인 또는 LLM이 외부 세계와 상호작용하기 위한 인터페이스
- Langchain에서 기본 제공하는 도구를 사용해서 쉽게 도구를 활용 가능 + 사용자 정의 도구를 쉽게 구축 가능

## Built-in tools

- tool : 단일 도구
- toolkit : 여러 도구를 묶어서 하나의 도구로 사용

## Python RERL 도구

- Python 코드를 REPL(Read-Eval-Print Loop) 환경에서 실행하기 위한 두 가지 주요 클래스 제공

### 설명

- Python 셸 환경 제공
- 유효한 Python 명령어를 입력으로 받아 실행
- 결과를 보기 위해 print 함수 사용

### 주요 특징

- sanitize_input : 입력을 정제하는 옵션
- python_repl : PythonREPL 인스턴스

### 사용 방법

- PythonREPLTool 인스턴스 생성
- run 또는 arun, invoke 메서드를 사용해서 Python 코드 실행

### 입력 정제

- 입력 문자열에서 불필요한 공백, 백텍, 'python' 키워드 등을 제거

### 흐름 정리

1. LLM 모델에게 특정 작업을 수행하는 Python 코드를 작성하도록 요청
2. 작성된 코드를 실행하여 결과 얻기
3. 결과 출력

## 검색 API 도구

- Tavily 검색 API를 활용해서 검색 기능을 구현하는 도구
    - `TavilySearchResults`, `TavilyAnswer`

### `TavilySearchResults`

- 설명
    - Tavily 검색 API를 쿼리하고 JSON 형식의 결과를 반환
    - 포괄적이고 정확하고 신뢰할 수 있는 결과에 최적화된 검색 엔진
    - 현재 이벤트에 대한 질문을 답변할 때 유용함.

## Image 생성 도구 (DALL-E)

- DallEAPIWrapper 클래스 : OpenAI의 DALL-E 이미지 생성기를 위한 래퍼 (wrapper)
    - 이 도구를 사용하면 DALL-E API를 쉽게 통합하여 텍스트 기반 이미지 생성 기능을 구현할 수 있음.

## 사용자 정의 도구

- 사용자가 직접 도구를 정의하여 사용할 수 있음.
- `langchain.tools` 모듈에서 제공하는 `tool` 데코레이터를 사용하여 함수를 도구로 변환

### `@tool` 데코레이터

- 설명
    - 함수를 도구로 변환하는 기능 제공
    - 다양한 옵션을 통해 도구의 동작을 커스터마이즈 할 수 있음.

- 사용 방법

1. 함수 위에 `@tool` 데코레이터 적용
2. 필요에 따라 데코레이터 매개변수 설정