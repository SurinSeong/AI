# AI 아카이브 시스템 설치 및 실행 가이드

## 프로젝트 구조
```
ai-archive-system/
├── app.py                # 메인 애플리케이션
├── requirements.txt      # 패키지 목록
├── archive.db           # SQLite 데이터베이스 (자동 생성)
└── test-image.png       # 테스트용 샘플 이미지
```

## 빠른 시작
```bash
# 의존성 설치
pip install -r requirements.txt

# 애플리케이션 실행
streamlit run app.py
```

## 사용 방법
1. http://localhost:8501 접속
2. **문서 업로드 탭**
   - PNG, JPG, JPEG 형식의 문서 이미지 업로드
   - 자동으로 문서 유형 분류 및 정보 추출
   - "저장" 버튼으로 데이터베이스에 보관
3. **문서 검색 탭**
   - 벡터 유사도 검색 또는 키워드 검색 선택
   - 검색어 입력 (예: "영수증", "계약서")
   - 검색 결과에서 이미지 미리보기 및 다운로드
4. **문서 목록 탭**
   - 저장된 모든 문서 목록 확인
   - 각 문서의 상세 정보 열람

## 주요 기능
- DiT 기반 문서 유형 자동 분류 
- PaddleOCR 한국어 텍스트 추출 및 위치 검출
- Donut 모델 영수증 정보 자동 구조화
- LayoutLMv3 문서 레이아웃 분석 및 정보 추출
- Ko-SRoBERTa 벡터 임베딩 기반 의미 검색
- KoBART 한국어 문서 자동 요약
- SQLModel + SQLite 문서 메타데이터 관리
- Streamlit 웹 기반 사용자 인터페이스

### [Streamlit](https://github.com/streamlit/streamlit)

- Python 기반 오픈소스 웹 어플리케이션 프레임워크
- Tornado 웹 서버와 React 기반의 프론트엔드로 구성되어 있음.
- Hot-reloading 기능을 지원하는 등 Python 기반으로 Web UI를 쉽고 빠르게 개발하게 해준다.

### [DiT (Document Image Transformer)](https://github.com/microsoft/unilm/tree/master/dit)

- Microsoft에서 개발한 문서 이미지 분류 모델로 Vision Transformer(ViT) 아키텍처를 기반으로 300,000개의 문서 이미지, 16개의 카테고리로 사전학습되었음.
- 이미지를 16x16으로 분할하여 토큰으로 전환하고 문서의 전역적 특징을 학습.

### [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

- 바이두에서 개발한 오픈소스 OCR 엔진
- 텍스트 검출, 방향 분류, 텍스트 인식의 3단계 파이프라인을 구현한다.
- 한국어를 포함한 80개 이상의 언어를 지원

### [LayoutLMv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3)

- Microsoft에서 개발한 멀티모달 문서 인식 모델
- 텍스트, 레이아웃(위치 정보), 이미지를 통합하여 처리
- OCR 처리 후 추출된 텍스트와 바운딩 박스 정보를 활용해 문서의 유형(문서, 영수증)에 따른 문서의 구조적 정보(제목, 본문, 날짜, 금액 등)를 추출하는 것에 사용

### [Donut (Document Understanding Transformer)](https://github.com/clovaai/donut)

- 네이버 클로바에서 개발한 OCR-free 문서 인식 모델
- 별도의 OCR 과정 없이 이미지에서 직접 구조화된 정보를 추출
- 영수증 데이터셋으로 Fine-tuning 되었고, 영문 모델로 LayoutLMv3를 보완하는 용도로만 사용함.

### [Ko-SRoBERTa](https://github.com/jhgan00/ko-sentence-transformers)

- 문장을 768차원의 밀집 벡터로 인코딩해서 의미적으로 유사한 문장은 코사인 유사도가 높도록 학습된 문장 임베딩 모델
- 문서의 요약과 키워드를 통해 유사한 의미를 가진 문서들을 검색하는 것에 사용됨.

