import json  # JSON 데이터 처리를 위한 모듈
from datetime import datetime  # 날짜와 시간 처리를 위한 모듈
from typing import List, Dict, Optional  # 타입 힌팅을 위한 모듈
from fastmcp import FastMCP  # MCP(Model Context Protocol) 서버 구현을 위한 라이브러리
from pathlib import Path  # 파일 경로 처리를 위한 모던한 방식의 라이브러리

# =============================================================================
# MCP 서버 초기화 및 데이터 설정
# =============================================================================

# MCP 서버 인스턴스 생성 - 서버의 이름을 "개인 메모 & 할 일 관리"로 설정
mcp = FastMCP("개인 메모 & 할 일 관리")

# 데이터 저장을 위한 디렉토리와 파일 경로 설정
# DATA_DIR = Path("data")  # 모든 데이터를 저장할 디렉토리

BASE_DIR = Path(__file__).resolve().parent  # 현재 파일(server.py)의 위치
DATA_DIR = BASE_DIR / "data"                # 그 하위에 'data' 폴더 지정

NOTES_FILE = DATA_DIR / "notes.json"  # 메모 데이터를 저장할 JSON 파일
TODOS_FILE = DATA_DIR / "todos.json"  # 할 일 데이터를 저장할 JSON 파일

# 데이터 디렉토리가 존재하지 않으면 생성
# exist_ok=True 옵션으로 이미 존재하는 경우 에러가 발생하지 않음
DATA_DIR.mkdir(exist_ok=True)

# =============================================================================
# 데이터 입출력 유틸리티 함수들
# =============================================================================

def load_notes() -> Dict[str, dict]:
    """
    저장된 메모 데이터를 JSON 파일에서 로드하는 함수
    
    Returns:
        Dict[str, dict]: 메모 ID를 키로 하고 메모 객체를 값으로 하는 딕셔너리
                        파일이 존재하지 않으면 빈 딕셔너리를 반환
    """
    # 메모 파일이 존재하는지 확인
    if NOTES_FILE.exists():
        # 파일이 존재하면 JSON 데이터를 읽어서 반환
        # encoding='utf-8'로 한글 문자열 처리를 보장
        with open(NOTES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    # 파일이 존재하지 않으면 빈 딕셔너리 반환
    return {}

def save_notes(notes: Dict[str, dict]):
    """
    메모 데이터를 JSON 파일에 저장하는 함수
    
    Args:
        notes (Dict[str, dict]): 저장할 메모 데이터 딕셔너리
    """
    # JSON 파일에 메모 데이터 저장
    with open(NOTES_FILE, 'w', encoding='utf-8') as f:
        json.dump(notes, f, 
                 ensure_ascii=False,  # 한글 등 유니코드 문자를 그대로 저장
                 indent=2)           # 들여쓰기로 가독성 향상

def load_todos() -> Dict[str, dict]:
    """
    저장된 할 일 데이터를 JSON 파일에서 로드하는 함수
    
    Returns:
        Dict[str, dict]: 할 일 ID를 키로 하고 할 일 객체를 값으로 하는 딕셔너리
                        파일이 존재하지 않으면 빈 딕셔너리를 반환
    """
    # 할 일 파일이 존재하는지 확인
    if TODOS_FILE.exists():
        # 파일이 존재하면 JSON 데이터를 읽어서 반환
        with open(TODOS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    # 파일이 존재하지 않으면 빈 딕셔너리 반환
    return {}

def save_todos(todos: Dict[str, dict]):
    """
    할 일 데이터를 JSON 파일에 저장하는 함수
    
    Args:
        todos (Dict[str, dict]): 저장할 할 일 데이터 딕셔너리
    """
    # JSON 파일에 할 일 데이터 저장
    with open(TODOS_FILE, 'w', encoding='utf-8') as f:
        json.dump(todos, f, 
                 ensure_ascii=False,  # 한글 등 유니코드 문자를 그대로 저장
                 indent=2)           # 들여쓰기로 가독성 향상

# =============================================================================
# 메모 관련 MCP 도구들
# =============================================================================

@mcp.tool()
def create_note(title: str, content: str, tags: List[str] = None) -> str:
    """
    새로운 메모를 생성하고 저장하는 MCP 도구
    
    Args:
        title (str): 메모의 제목
        content (str): 메모의 내용
        tags (List[str], optional): 메모에 붙일 태그 목록. 기본값은 None
    
    Returns:
        str: 메모 생성 성공 메시지와 생성된 메모 ID
    """
    # 기존 메모 데이터 로드
    notes = load_notes()
    
    # 현재 날짜와 시간을 기반으로 고유한 메모 ID 생성
    # 형식: note_YYYYMMDD_HHMMSS (예: note_20231215_143022)
    note_id = f"note_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 새 메모 객체 생성 - 메모의 모든 정보를 포함하는 딕셔너리
    note = {
        "id": note_id,                           # 메모의 고유 식별자
        "title": title,                          # 메모 제목
        "content": content,                      # 메모 내용
        "tags": tags or [],                      # 태그 목록 (None인 경우 빈 리스트)
        "created_at": datetime.now().isoformat(), # 생성 시간 (ISO 형식)
        "updated_at": datetime.now().isoformat()  # 최종 수정 시간 (ISO 형식)
    }
    
    # 메모 딕셔너리에 새 메모 추가
    notes[note_id] = note
    
    # 변경된 메모 데이터를 파일에 저장
    save_notes(notes)
    
    # 성공 메시지 반환
    return f"메모가 생성되었습니다! (ID: {note_id})"

@mcp.tool()
def list_notes(tag: Optional[str] = None) -> List[dict]:
    """
    저장된 메모 목록을 조회하는 MCP 도구
    
    Args:
        tag (Optional[str]): 특정 태그로 필터링할 태그명. None이면 모든 메모 반환
    
    Returns:
        List[dict]: 메모 객체들의 리스트 (최신순으로 정렬됨)
    """
    # 저장된 모든 메모 데이터 로드
    notes = load_notes()
    
    # 태그 필터링 처리
    if tag:
        # 지정된 태그를 포함하는 메모만 필터링
        # v.get('tags', [])로 tags 키가 없는 경우 빈 리스트를 기본값으로 사용
        filtered_notes = {
            k: v for k, v in notes.items() 
            if tag in v.get('tags', [])
        }
    else:
        # 태그 필터가 없으면 모든 메모 반환
        filtered_notes = notes
    
    # 딕셔너리를 리스트로 변환 (메모 객체들만 추출)
    notes_list = list(filtered_notes.values())
    
    # 생성일 기준으로 최신순 정렬 (reverse=True로 내림차순)
    notes_list.sort(key=lambda x: x['created_at'], reverse=True)
    
    return notes_list

@mcp.tool()
def search_notes(query: str) -> List[dict]:
    """
    메모를 제목과 내용에서 검색하는 MCP 도구
    
    Args:
        query (str): 검색할 키워드
    
    Returns:
        List[dict]: 검색 조건에 맞는 메모 객체들의 리스트 (최신순으로 정렬됨)
    """
    # 저장된 모든 메모 데이터 로드
    notes = load_notes()
    
    # 검색어를 소문자로 변환하여 대소문자 구분 없이 검색
    query_lower = query.lower()
    
    # 검색 결과를 저장할 리스트
    results = []
    
    # 모든 메모를 순회하며 검색
    for note in notes.values():
        # 제목이나 내용에 검색어가 포함되어 있는지 확인
        # 대소문자 구분 없이 검색하기 위해 모두 소문자로 변환하여 비교
        if (query_lower in note['title'].lower() or 
            query_lower in note['content'].lower()):
            results.append(note)
    
    # 검색 결과를 생성일 기준으로 최신순 정렬
    results.sort(key=lambda x: x['created_at'], reverse=True)
    
    return results

@mcp.tool()
def update_note(note_id: str, title: Optional[str] = None, 
                content: Optional[str] = None, tags: Optional[List[str]] = None) -> str:
    """
    기존 메모를 수정하는 MCP 도구
    
    Args:
        note_id (str): 수정할 메모의 ID
        title (Optional[str]): 새로운 제목 (변경하지 않으려면 None)
        content (Optional[str]): 새로운 내용 (변경하지 않으려면 None)
        tags (Optional[List[str]]): 새로운 태그 목록 (변경하지 않으려면 None)
    
    Returns:
        str: 수정 결과 메시지
    """
    # 저장된 모든 메모 데이터 로드
    notes = load_notes()
    
    # 지정된 ID의 메모가 존재하는지 확인
    if note_id not in notes:
        return f"메모를 찾을 수 없습니다: {note_id}"
    
    # 수정할 메모 객체 가져오기
    note = notes[note_id]
    
    # 전달된 매개변수가 None이 아닌 경우에만 해당 필드 업데이트
    # 이렇게 하면 사용자가 원하는 필드만 선택적으로 수정 가능
    if title is not None:
        note['title'] = title
    if content is not None:
        note['content'] = content
    if tags is not None:
        note['tags'] = tags
    
    # 수정 시간을 현재 시간으로 업데이트
    note['updated_at'] = datetime.now().isoformat()
    
    # 변경된 메모 데이터를 파일에 저장
    save_notes(notes)
    
    return f"메모가 수정되었습니다: {note_id}"

@mcp.tool()
def delete_note(note_id: str) -> str:
    """
    메모를 삭제하는 MCP 도구
    
    Args:
        note_id (str): 삭제할 메모의 ID
    
    Returns:
        str: 삭제 결과 메시지
    """
    # 저장된 모든 메모 데이터 로드
    notes = load_notes()
    
    # 지정된 ID의 메모가 존재하는지 확인
    if note_id not in notes:
        return f"메모를 찾을 수 없습니다: {note_id}"
    
    # 딕셔너리에서 해당 메모 삭제
    del notes[note_id]
    
    # 변경된 메모 데이터를 파일에 저장
    save_notes(notes)
    
    return f"메모가 삭제되었습니다: {note_id}"

# =============================================================================
# 할 일 관련 MCP 도구들
# =============================================================================

@mcp.tool()
def create_todo(title: str, description: str = "", 
                due_date: Optional[str] = None, priority: str = "medium") -> str:
    """
    새로운 할 일을 생성하고 저장하는 MCP 도구
    
    Args:
        title (str): 할 일의 제목
        description (str): 할 일의 상세 설명 (기본값: 빈 문자열)
        due_date (Optional[str]): 마감일 (YYYY-MM-DD 형식, 예: "2023-12-25")
        priority (str): 우선순위 ("low", "medium", "high" 중 하나, 기본값: "medium")
    
    Returns:
        str: 할 일 생성 성공 메시지와 생성된 할 일 ID
    """
    # 기존 할 일 데이터 로드
    todos = load_todos()
    
    # 현재 날짜와 시간을 기반으로 고유한 할 일 ID 생성
    # 형식: todo_YYYYMMDD_HHMMSS (예: todo_20231215_143022)
    todo_id = f"todo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 새 할 일 객체 생성 - 할 일의 모든 정보를 포함하는 딕셔너리
    todo = {
        "id": todo_id,                           # 할 일의 고유 식별자
        "title": title,                          # 할 일 제목
        "description": description,              # 할 일 상세 설명
        "completed": False,                      # 완료 상태 (기본값: 미완료)
        "priority": priority,                    # 우선순위
        "due_date": due_date,                    # 마감일 (None 가능)
        "created_at": datetime.now().isoformat(), # 생성 시간 (ISO 형식)
        "completed_at": None                     # 완료 시간 (완료되지 않았으므로 None)
    }
    
    # 할 일 딕셔너리에 새 할 일 추가
    todos[todo_id] = todo
    
    # 변경된 할 일 데이터를 파일에 저장
    save_todos(todos)
    
    # 성공 메시지 반환
    return f"할 일이 생성되었습니다! (ID: {todo_id})"

@mcp.tool()
def list_todos(status: str = "all", sort_by: str = "created") -> List[dict]:
    """
    저장된 할 일 목록을 조회하는 MCP 도구
    
    Args:
        status (str): 상태 필터 ("all", "pending", "completed" 중 하나, 기본값: "all")
        sort_by (str): 정렬 기준 ("created", "due_date", "priority" 중 하나, 기본값: "created")
    
    Returns:
        List[dict]: 할 일 객체들의 리스트 (지정된 기준으로 정렬됨)
    """
    # 저장된 모든 할 일 데이터 로드
    todos = load_todos()
    
    # 상태에 따른 필터링 처리
    if status == "pending":
        # 미완료 할 일만 필터링 (completed가 False인 것들)
        filtered_todos = {k: v for k, v in todos.items() if not v['completed']}
    elif status == "completed":
        # 완료된 할 일만 필터링 (completed가 True인 것들)
        filtered_todos = {k: v for k, v in todos.items() if v['completed']}
    else:
        # "all"인 경우 모든 할 일 반환
        filtered_todos = todos
    
    # 딕셔너리를 리스트로 변환 (할 일 객체들만 추출)
    todos_list = list(filtered_todos.values())
    
    # 지정된 기준에 따라 정렬 처리
    if sort_by == "due_date":
        # 마감일 기준 정렬
        # 마감일이 없는 항목(None)은 '9999-12-31'로 처리하여 맨 뒤로 보냄
        todos_list.sort(key=lambda x: x.get('due_date') or '9999-12-31')
    elif sort_by == "priority":
        # 우선순위 기준 정렬 (high -> medium -> low 순서)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        # get 메서드로 알 수 없는 우선순위는 medium(1)으로 처리
        todos_list.sort(key=lambda x: priority_order.get(x['priority'], 1))
    else:  # sort_by == "created"
        # 생성일 기준으로 최신순 정렬 (기본값)
        todos_list.sort(key=lambda x: x['created_at'], reverse=True)
    
    return todos_list

@mcp.tool()
def complete_todo(todo_id: str) -> str:
    """
    할 일을 완료 상태로 변경하는 MCP 도구
    
    Args:
        todo_id (str): 완료 처리할 할 일의 ID
    
    Returns:
        str: 완료 처리 결과 메시지
    """
    # 저장된 모든 할 일 데이터 로드
    todos = load_todos()
    
    # 지정된 ID의 할 일이 존재하는지 확인
    if todo_id not in todos:
        return f"할 일을 찾을 수 없습니다: {todo_id}"
    
    # 완료 처리할 할 일 객체 가져오기
    todo = todos[todo_id]
    
    # 이미 완료된 할 일인지 확인
    if todo['completed']:
        return f"이미 완료된 할 일입니다: {todo_id}"
    
    # 할 일을 완료 상태로 변경
    todo['completed'] = True
    # 완료 시간을 현재 시간으로 설정
    todo['completed_at'] = datetime.now().isoformat()
    
    # 변경된 할 일 데이터를 파일에 저장
    save_todos(todos)
    
    # 성공 메시지 반환 (할 일 제목도 함께 표시)
    return f"할 일이 완료되었습니다: {todo['title']}"

@mcp.tool()
def update_todo(todo_id: str, title: Optional[str] = None,
                description: Optional[str] = None,
                due_date: Optional[str] = None,
                priority: Optional[str] = None) -> str:
    """
    기존 할 일을 수정하는 MCP 도구
    
    Args:
        todo_id (str): 수정할 할 일의 ID
        title (Optional[str]): 새로운 제목 (변경하지 않으려면 None)
        description (Optional[str]): 새로운 설명 (변경하지 않으려면 None)
        due_date (Optional[str]): 새로운 마감일 (변경하지 않으려면 None)
        priority (Optional[str]): 새로운 우선순위 (변경하지 않으려면 None)
    
    Returns:
        str: 수정 결과 메시지
    """
    # 저장된 모든 할 일 데이터 로드
    todos = load_todos()
    
    # 지정된 ID의 할 일이 존재하는지 확인
    if todo_id not in todos:
        return f"할 일을 찾을 수 없습니다: {todo_id}"
    
    # 수정할 할 일 객체 가져오기
    todo = todos[todo_id]
    
    # 전달된 매개변수가 None이 아닌 경우에만 해당 필드 업데이트
    # 이렇게 하면 사용자가 원하는 필드만 선택적으로 수정 가능
    if title is not None:
        todo['title'] = title
    if description is not None:
        todo['description'] = description
    if due_date is not None:
        todo['due_date'] = due_date
    if priority is not None:
        todo['priority'] = priority
    
    # 변경된 할 일 데이터를 파일에 저장
    save_todos(todos)
    
    return f"할 일이 수정되었습니다: {todo_id}"

@mcp.tool()
def delete_todo(todo_id: str) -> str:
    """
    할 일을 삭제하는 MCP 도구
    
    Args:
        todo_id (str): 삭제할 할 일의 ID
    
    Returns:
        str: 삭제 결과 메시지
    """
    # 저장된 모든 할 일 데이터 로드
    todos = load_todos()
    
    # 지정된 ID의 할 일이 존재하는지 확인
    if todo_id not in todos:
        return f"할 일을 찾을 수 없습니다: {todo_id}"
    
    # 딕셔너리에서 해당 할 일 삭제
    del todos[todo_id]
    
    # 변경된 할 일 데이터를 파일에 저장
    save_todos(todos)
    
    return f"할 일이 삭제되었습니다: {todo_id}"

# =============================================================================
# 통계 및 리포트 MCP 도구
# =============================================================================

@mcp.tool()
def get_statistics() -> dict:
    """
    메모와 할 일의 전체 통계 정보를 제공하는 MCP 도구
    
    Returns:
        dict: 메모와 할 일의 상세 통계 정보를 포함하는 딕셔너리
    """
    # 저장된 메모와 할 일 데이터 로드
    notes = load_notes()
    todos = load_todos()
    
    # 할 일 완료 통계 계산
    # 완료된 할 일 개수 세기 (completed가 True인 것들)
    completed_todos = sum(1 for t in todos.values() if t['completed'])
    # 미완료 할 일 개수 = 전체 할 일 - 완료된 할 일
    pending_todos = len(todos) - completed_todos
    
    # 미완료 할 일의 우선순위별 통계 계산
    priority_stats = {"high": 0, "medium": 0, "low": 0}
    for todo in todos.values():
        # 완료되지 않은 할 일만 대상으로 함
        if not todo['completed']:
            # 우선순위가 설정되지 않은 경우 'medium'을 기본값으로 사용
            priority = todo.get('priority', 'medium')
            priority_stats[priority] += 1
    
    # 메모의 태그 사용 통계 계산
    all_tags = []  # 모든 태그를 수집할 리스트
    for note in notes.values():
        # 각 메모의 태그들을 전체 태그 리스트에 추가
        # get('tags', [])로 태그가 없는 메모는 빈 리스트로 처리
        all_tags.extend(note.get('tags', []))
    
    # 태그별 사용 횟수 계산
    tag_counts = {}
    for tag in all_tags:
        # 각 태그의 사용 횟수를 카운트
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    # 통계 정보를 구조화된 딕셔너리로 반환
    return {
        "notes": {
            "total": len(notes),        # 전체 메모 개수
            "tags": tag_counts          # 태그별 사용 횟수
        },
        "todos": {
            "total": len(todos),                    # 전체 할 일 개수
            "completed": completed_todos,           # 완료된 할 일 개수
            "pending": pending_todos,               # 미완료 할 일 개수
            "priority_breakdown": priority_stats    # 우선순위별 미완료 할 일 개수
        },
        "last_updated": datetime.now().isoformat()  # 통계 생성 시간
    }

# =============================================================================
# MCP 리소스 제공 설정
# =============================================================================

# =============================================================================
# MCP 리소스 제공 설정
# =============================================================================

@mcp.resource("notes://all")
def get_all_notes():
    """
    모든 메모를 MCP 리소스로 제공하는 함수
    
    클라이언트가 "notes://all" 리소스를 요청할 때 호출됨
    
    Returns:
        List[dict]: 모든 메모의 리스트
    """
    # MCP 도구가 아닌 직접 데이터 로드 함수 호출
    notes = load_notes()
    notes_list = list(notes.values())
    notes_list.sort(key=lambda x: x['created_at'], reverse=True)
    return notes_list

@mcp.resource("todos://pending")
def get_pending_todos():
    """
    미완료 할일을 MCP 리소스로 제공하는 함수
    
    클라이언트가 "todos://pending" 리소스를 요청할 때 호출됨
    
    Returns:
        List[dict]: 미완료 할일의 리스트
    """
    # MCP 도구가 아닌 직접 데이터 로드 함수 호출
    todos = load_todos()
    filtered_todos = {k: v for k, v in todos.items() if not v['completed']}
    todos_list = list(filtered_todos.values())
    todos_list.sort(key=lambda x: x['created_at'], reverse=True)
    return todos_list

# =============================================================================
# 메인 실행 부분
# =============================================================================

if __name__ == "__main__":
    # 서버 시작 메시지 출력
    print("개인 메모 & 할 일 관리 MCP 서버를 시작합니다...")
    # 데이터 저장 위치 정보 출력 (절대 경로로 표시)
    print(f"데이터 저장 위치: {DATA_DIR.absolute()}")
    
    # MCP 서버 실행
    # 이 함수는 서버를 시작하고 클라이언트의 요청을 대기함
    mcp.run()