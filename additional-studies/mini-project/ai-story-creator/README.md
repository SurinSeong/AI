# AI 스토리북 저작 도구 설치 및 실행 가이드

## 프로젝트 구조
```
ai-storybook-creator/
├── app.py                  # 메인 애플리케이션
├── requirements.txt        # 패키지 목록
├── storybook.db           # SQLite 데이터베이스 (자동 생성)
```

## 빠른 시작
```bash
# 수동 실행
pip install -r requirements.txt
python app.py
```

## 사용 방법
1. http://localhost:7860 접속
2. 스토리 주제 입력 (예: "스타트업 창업 성공 스토리")
3. "스토리 생성" 클릭
4. 생성된 스토리 확인 및 수정
5. "스토리북 생성" 클릭하여 이미지와 PDF 생성

## 주요 기능
- Bllossom LLM 기반 스토리 생성 (5개 문단)
- DreamShaper 모델로 문단별 삽화 자동 생성
- 한국어 텍스트 분석 → 영어 프롬프트 변환
- 이미지 해시 기반 캐싱 시스템
- ReportLab을 활용한 PDF 스토리북 제작
- SQLModel 기반 스토리/이미지 이력 관리
- Gradio 웹 기반 사용자 인터페이스