# Introducing the Model Context Protocol

- 핵심 : Protocol
    - HTTP 통신, API

[teddy-note MCP usecase](https://github.com/teddynote-lab/mcp-usecase/tree/main)

## Agent의 구조

- LLM (핵심) - GPT, Claude, Gemini
- **Tool** : **외부 tool 사용** 가능 ⇒ **MCP 담당** 역할!
    - 기존 LLM에서 능력 확장 가능
- **React (Framework)**
    - Thought - Action - Conservation 반복 → 문제 스스로 해결

## 왜 갑자기 관심이 증가했을까?

- **Cursor AI에 MCP 업데이트**!!
    - Langchain에 이미 다양한 많은 tool을 가지고 있음. ⇒ 일부 개발자만 사용한다.
        - 랭체인에서만 호환 가능
    - 독립적인 tool을 공개함. 하지만 인기 딱히..
    - 하지만, Cursor AI를 몸으로 하고 MCP를 팔, 다리로 해서 **코딩할 수 없는 사람도 사용**할 수 있도록 했음.
        - = 호환성이 좋음.
    - 사용자가 많은 Cursor AI가 Integrate함으로써 많은 사람의 관심을 받음.
    - 앱스토어[smithery](https://smithery.ai/) 에 사용할 수 있는 tool들이 계속해서 늘어나게 되며 선순환이 된다.
    - 곧 openAI도 사용할 수 있도록 할 것 같음..
    - langchain에서도 **mcp를 지원**하는 프로젝트를 많이 내고 있음.

### 간단하게 개발도 가능함!

[MCP](https://github.com/modelcontextprotocol/python-sdk)

## UseCase + 구현해보기

1. RAG (Local)
    - 랭체인 코드로 RAG를 만들어두고, MCP 서버로 띄우기
    - →  CursorAI가  Langchain으로 만든 RAG 시스템 참조
    - → Claude Desktop에서도 참조 가능하다.
    - Front 구현 필요가 없음.. 왜냐면 Claude에서 RAG를 사용할 수 있으니깐..
2. Dify
    - external knowledge API 참조
3. Dify workflow
    - 구축해놓은 workflow 호출하기
4. Python Function
    - custom logic 생성 → 호출

### Langchain & LangGraph

[langgraph-mcp-agents](https://github.com/teddynote-lab/langgraph-mcp-agents/)
