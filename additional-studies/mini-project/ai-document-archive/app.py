import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForImageClassification, AutoProcessor, 
    AutoTokenizer, AutoModelForSeq2SeqLM,
    VisionEncoderDecoderModel, LayoutLMv3Processor, LayoutLMv3ForTokenClassification,
)
from paddleocr import PaddleOCR
from sqlmodel import Field, Session, SQLModel, create_engine, select
from datetime import datetime
from PIL import Image
import io
import base64
import numpy as np
from typing import Optional
import json
import re
from sklearn.metrics.pairwise import cosine_similarity

# 데이터베이스 모델
class Document(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    doc_type: str
    content: str
    summary: str
    keywords: str
    structured_data: str  # JSON 형태로 저장
    upload_date: datetime = Field(default_factory=datetime.now)
    image_data: bytes
    embedding: Optional[str] = None  # 벡터를 JSON으로 저장

# 데이터베이스 초기화
engine = create_engine("sqlite:///archive.db")
SQLModel.metadata.create_all(engine)

# 모델 로드
@st.cache_resource
def load_models():
    # DiT 문서 분류
    dit_processor = AutoProcessor.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
    dit_model = AutoModelForImageClassification.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
    
    # OCR
    ocr = PaddleOCR(lang='korean')
    
    # Donut (영수증 전용)
    donut_processor = AutoProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
    donut_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
    
    # LayoutLMv3
    layout_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    layout_model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")
    
    # 텍스트 요약
    summarizer_tokenizer = AutoTokenizer.from_pretrained("gangyeolkim/kobart-korean-summarizer-v2")
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("gangyeolkim/kobart-korean-summarizer-v2")
    
    # 임베딩 모델 (벡터 검색용)
    embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    
    return (dit_processor, dit_model, ocr, donut_processor, donut_model, 
            layout_processor, layout_model, summarizer_tokenizer, summarizer_model,
            embedding_model)

# 문서 유형 분류
def classify_document(image, dit_processor, dit_model):
    inputs = dit_processor(images=image, return_tensors="pt")
    outputs = dit_model(**inputs)
    predicted_class_idx = outputs.logits.argmax(-1).item()
    predicted_class = dit_model.config.id2label[predicted_class_idx]
    
    # 영수증 관련 클래스 매핑
    if any(keyword in predicted_class.lower() for keyword in ['invoice']):
        return "영수증"
    return predicted_class

# 영수증 OCR (Donut)
def extract_receipt_info(image, processor, model):
    pixel_values = processor(image, return_tensors="pt").pixel_values
    decoder_input_ids = processor.tokenizer("<s_cord-v2>", return_tensors="pt").input_ids
    
    outputs = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=512)
    prediction = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # JSON 파싱
    try:
        receipt_data = json.loads(prediction)
        return receipt_data
    except:
        return {}

# 일반 OCR
def extract_text_with_layout(image, ocr):
    result = PaddleOCR(lang='korean').ocr(np.array(image))
    text = ""
    boxes = []
    
    if result[0]:
        for line in result[0]:
            text += line[1][0] + " "
            boxes.append(line[0])
    
    return text.strip(), boxes


# LayoutLMv3를 활용한 구조화된 정보 추출 함수
def extract_structured_with_layoutlm(image, text, boxes, layout_processor, layout_model, doc_type):
    """
    LayoutLMv3를 사용하여 문서의 구조적 정보를 추출합니다.
    주의: 기본 LayoutLMv3는 특정 태스크에 fine-tuning되지 않았으므로,
    임베딩과 위치 정보를 활용한 휴리스틱 방식을 사용합니다.
    
    Args:
        image: PIL Image 객체
        text: OCR로 추출한 텍스트
        boxes: OCR로 추출한 바운딩 박스 좌표
        layout_processor: LayoutLMv3 프로세서
        layout_model: LayoutLMv3 모델
        doc_type: 문서 유형 (영수증, 문서 등)
    
    Returns:
        structured_data: 추출된 구조화된 정보 딕셔너리
    """
    # OCR 결과를 단어 단위로 분리
    words = text.split()
    
    # 빈 문자열이거나 단어가 없는 경우 처리
    if not words:
        return {}
    
    # 바운딩 박스가 없거나 부족한 경우 처리
    if not boxes or len(boxes) == 0:
        # 박스 정보 없이 기본 구조만 반환
        if doc_type == "영수증":
            return {"텍스트": text}
        else:
            return {"내용": text}
    
    # 바운딩 박스가 단어 수보다 적을 경우 처리
    if len(boxes) < len(words):
        # 마지막 박스를 복사하여 맞춤
        boxes = boxes + [boxes[-1]] * (len(words) - len(boxes))
    
    # 바운딩 박스를 정규화 (0-1000 범위로)
    width, height = image.size
    normalized_boxes = []
    word_positions = []  # 각 단어의 위치 정보 저장
    
    for idx, box in enumerate(boxes[:len(words)]):
        # box는 [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] 형태
        if len(box) >= 4:
            x1, y1 = box[0]
            x2, y2 = box[2]
            # LayoutLM은 [x1, y1, x2, y2] 형태로 0-1000 범위 좌표 필요
            norm_box = [
                int(x1 * 1000 / width),
                int(y1 * 1000 / height),
                int(x2 * 1000 / width),
                int(y2 * 1000 / height)
            ]
            normalized_boxes.append(norm_box)
            
            # 위치 정보 저장 (상대적 위치)
            word_positions.append({
                'word': words[idx] if idx < len(words) else '',
                'x_center': (norm_box[0] + norm_box[2]) / 2,
                'y_center': (norm_box[1] + norm_box[3]) / 2,
                'width': norm_box[2] - norm_box[0],
                'height': norm_box[3] - norm_box[1],
                'area': (norm_box[2] - norm_box[0]) * (norm_box[3] - norm_box[1])
            })
    
    # LayoutLMv3 입력 준비
    encoding = layout_processor(
        image,
        words[:len(normalized_boxes)],
        boxes=normalized_boxes,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    
    # 모델 추론 - 임베딩 추출
    with torch.no_grad():
        outputs = layout_model(**encoding, output_hidden_states=True)
    
    # 마지막 은닉층의 임베딩 사용
    last_hidden_states = outputs.hidden_states[-1]
    
    # 문서 유형별 구조화된 정보 추출
    if doc_type == "영수증":
        structured_data = extract_receipt_structure(word_positions, words, last_hidden_states)
    else:
        structured_data = extract_document_structure(word_positions, words, last_hidden_states)
    
    return structured_data


def extract_receipt_structure(word_positions, words, embeddings):
    """
    영수증 문서에서 구조화된 정보를 추출합니다.
    위치 정보와 텍스트 패턴을 활용한 휴리스틱 방식을 사용합니다.
    """
    structured_data = {
        "상호명": None,
        "날짜": None,
        "시간": None,
        "품목": [],
        "금액": [],
        "합계": None,
        "주소": None,
        "전화번호": None,
        "사업자번호": None
    }
    
    # 위치별로 단어 그룹화
    # 상단 20% - 주로 상호명, 주소
    # 중간 60% - 품목과 금액
    # 하단 20% - 합계, 사업자번호 등
    
    top_words = []
    middle_words = []
    bottom_words = []
    
    for pos in word_positions:
        if pos['y_center'] < 200:  # 상단
            top_words.append(pos)
        elif pos['y_center'] < 800:  # 중간
            middle_words.append(pos)
        else:  # 하단
            bottom_words.append(pos)
    
    # 1. 상호명 추출 (상단에서 가장 큰 텍스트)
    if top_words:
        # 면적이 가장 큰 텍스트를 상호명으로 추정
        largest = max(top_words, key=lambda x: x['area'])
        structured_data["상호명"] = largest['word']
    
    # 2. 날짜와 시간 추출
    for pos in word_positions:
        word = pos['word']
        # 날짜 패턴
        date_match = re.search(r'(\d{4})[.-](\d{1,2})[.-](\d{1,2})', word)
        if date_match:
            structured_data["날짜"] = word
        
        # 시간 패턴
        time_match = re.search(r'(\d{1,2}):(\d{2})', word)
        if time_match:
            structured_data["시간"] = word
    
    # 3. 금액 추출 (중간 영역에서 우측에 위치한 숫자)
    amount_pattern = r'[\d,]+원?'
    for pos in middle_words:
        word = pos['word']
        if re.match(amount_pattern, word) and pos['x_center'] > 500:  # 우측
            # 숫자만 추출 (콤마, 원, 기타 문자 제거)
            cleaned_amount = re.sub(r'[^\d]', '', word)
            if cleaned_amount and cleaned_amount.isdigit():
                structured_data["금액"].append(cleaned_amount)
    
    # 4. 품목 추출 (중간 영역에서 좌측에 위치한 텍스트)
    for pos in middle_words:
        word = pos['word']
        if pos['x_center'] < 500 and not re.match(r'[\d,]+', word):  # 좌측, 숫자 아님
            structured_data["품목"].append(word)
    
    # 5. 합계 추출 (하단에서 가장 큰 금액)
    total_candidates = []
    for pos in bottom_words:
        word = pos['word']
        if '합계' in word or '총' in word:
            # 합계 키워드 근처의 금액 찾기
            idx = word_positions.index(pos)
            for i in range(max(0, idx-3), min(len(word_positions), idx+4)):
                nearby_word = word_positions[i]['word']
                if re.match(r'[\d,]+원?', nearby_word):
                    # 숫자만 추출
                    cleaned_amount = re.sub(r'[^\d]', '', nearby_word)
                    if cleaned_amount and cleaned_amount.isdigit():
                        amount = int(cleaned_amount)
                        total_candidates.append(amount)
    
    if total_candidates:
        structured_data["합계"] = str(max(total_candidates))
    elif structured_data["금액"]:
        # 합계가 명시되지 않은 경우 금액 중 최대값
        try:
            amounts = []
            for x in structured_data["금액"]:
                if x.isdigit():
                    amounts.append(int(x))
            if amounts:
                structured_data["합계"] = str(max(amounts))
        except (ValueError, AttributeError):
            pass  # 변환 실패 시 합계 생략
    
    # 6. 사업자번호 추출
    for word in words:
        # 사업자번호 패턴 (XXX-XX-XXXXX)
        business_match = re.search(r'\d{3}-\d{2}-\d{5}', word)
        if business_match:
            structured_data["사업자번호"] = business_match.group()
    
    # 7. 전화번호 추출
    for word in words:
        # 전화번호 패턴
        phone_match = re.search(r'(\d{2,3})-(\d{3,4})-(\d{4})', word)
        if phone_match:
            structured_data["전화번호"] = phone_match.group()
    
    # None 값 제거
    structured_data = {k: v for k, v in structured_data.items() if v}
    
    return structured_data


def extract_document_structure(word_positions, words, embeddings):
    """
    일반 문서에서 구조화된 정보를 추출합니다.
    """
    structured_data = {
        "제목": None,
        "부제목": [],
        "날짜": None,
        "작성자": None,
        "핵심내용": [],
        "번호정보": []  # 전화번호, 계좌번호 등
    }
    
    # 1. 제목 추출 (상단에서 가장 큰 텍스트)
    top_words = [pos for pos in word_positions if pos['y_center'] < 200]
    if top_words:
        largest = max(top_words, key=lambda x: x['area'])
        structured_data["제목"] = largest['word']
    
    # 2. 부제목 추출 (중간 크기의 텍스트)
    avg_area = sum(pos['area'] for pos in word_positions) / len(word_positions)
    for pos in word_positions:
        if pos['area'] > avg_area * 1.5 and pos['word'] != structured_data.get("제목"):
            structured_data["부제목"].append(pos['word'])
    
    # 3. 날짜 추출
    for word in words:
        date_match = re.search(r'(\d{4})[.-](\d{1,2})[.-](\d{1,2})', word)
        if date_match:
            structured_data["날짜"] = date_match.group()
            break
    
    # 4. 번호 정보 추출
    for word in words:
        # 전화번호
        phone_match = re.search(r'(\d{2,3})-(\d{3,4})-(\d{4})', word)
        if phone_match:
            structured_data["번호정보"].append(f"전화: {phone_match.group()}")
        
        # 계좌번호 패턴 (숫자가 10자리 이상)
        account_match = re.search(r'\d{10,}', word)
        if account_match:
            structured_data["번호정보"].append(f"계좌: {account_match.group()}")
    
    # 5. 핵심 내용 추출 (텍스트 길이가 긴 부분)
    long_texts = []
    current_text = []
    
    for i, word in enumerate(words):
        current_text.append(word)
        # 문장 종료 판단
        if word.endswith('.') or word.endswith('!') or word.endswith('?') or i == len(words) - 1:
            sentence = ' '.join(current_text)
            if len(sentence) > 20:  # 20자 이상인 문장
                long_texts.append(sentence)
            current_text = []
    
    # 가장 긴 3개 문장을 핵심 내용으로
    structured_data["핵심내용"] = sorted(long_texts, key=len, reverse=True)[:3]
    
    # None 또는 빈 리스트 제거
    structured_data = {k: v for k, v in structured_data.items() if v}
    
    return structured_data


# 구조화된 정보 추출
def extract_structured_info(text, doc_type):
    structured_data = {}
    
    if doc_type == "영수증":
        # 날짜 추출
        date_pattern = r'(\d{4})[.-](\d{1,2})[.-](\d{1,2})'
        date_match = re.search(date_pattern, text)
        if date_match:
            structured_data['date'] = f"{date_match.group(1)}.{date_match.group(2)}.{date_match.group(3)}"
        
        # 금액 추출
        price_pattern = r'([0-9,]+)\s*원'
        prices = re.findall(price_pattern, text)
        if prices:
            structured_data['amounts'] = [p.replace(',', '') for p in prices]
            structured_data['total'] = max([int(p.replace(',', '')) for p in prices])
        
        # 상호명 추출 (간단한 휴리스틱)
        words = text.split()
        if len(words) > 0:
            structured_data['store'] = words[0]
    
    return structured_data

# 텍스트 요약
def summarize_text(text, tokenizer, model):
    if not text or len(text) < 50:
        return text
    
    inputs = tokenizer(text[:1024], return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=128, min_length=20, num_beams=4)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# 텍스트 임베딩 생성
def create_embedding(text, model):
    if not text or len(text.strip()) < 2:
        return None
    
    # 직접 문장 임베딩 생성 (훨씬 간단하고 효과적)
    embedding = model.encode(text)
    return embedding.tolist()

# 키워드 추출
def extract_keywords(text, structured_data=None):
    stopwords = ['은', '는', '이', '가', '을', '를', '의', '에', '와', '과', '에서', '으로']
    words = text.split()
    keywords = [w for w in words if len(w) > 1 and w not in stopwords]
    
    # 구조화된 데이터에서 추가 키워드
    if structured_data:
        if 'store' in structured_data:
            keywords.append(structured_data['store'])
        if 'date' in structured_data:
            keywords.append(structured_data['date'])
    
    return ", ".join(list(set(keywords)))

# 문서 처리
def process_document(uploaded_file, models):
    (dit_processor, dit_model, ocr, donut_processor, donut_model, 
     layout_processor, layout_model, sum_tokenizer, sum_model,
     embedding_model) = models
    
    image = Image.open(uploaded_file).convert("RGB")
    
    # 1. 문서 유형 분류
    doc_type = classify_document(image, dit_processor, dit_model)
    print('\n==========')
    print(f'\n[doc_type]\n{doc_type}')
    content, boxes = extract_text_with_layout(image, ocr)
    layoutlm_data = extract_structured_with_layoutlm(image, content, boxes, layout_processor, layout_model, doc_type)
    print('\n==========')
    print(f'\n[layoutlm_data]\n{layoutlm_data}')
    # 2. 문서 유형별 처리
    if doc_type == "영수증":
        print('영수증')
        # Donut으로 영수증 정보 추출
        receipt_info = extract_receipt_info(image, donut_processor, donut_model)
        print('\n==========')
        print(f'\n[receipt_info]\n{receipt_info}')
        
        # 구조화된 정보 병합
        structured_data = extract_structured_info(content, doc_type)
        if receipt_info:
            # Donut 결과가 있으면 LayoutML과 병합    
            for key, value in receipt_info.items():
                if key not in layoutlm_data or not layoutlm_data[key]:
                    layoutlm_data[key] = value
        else:
            structured_data = layoutlm_data
        structured_data.update(receipt_info)
    else:
        # 일반 OCR
        structured_data = layoutlm_data

    print('\n==========')
    print(f'\n[content]\n{content}')
    print('\n==========')
    print(f'\n[structured_data]\n{structured_data}')
    
    # 3. 요약 생성
    summary = summarize_text(content, sum_tokenizer, sum_model)

    print('\n==========')
    print(f'\n[summary]\n{summary}')
    
    # 4. 키워드 추출
    keywords = extract_keywords(content, structured_data)
    print('\n==========')
    print(f'\n[keywords]\n{keywords}')
    
    # 5. 임베딩 생성
    embedding = create_embedding(content + " " + summary, embedding_model)
    
    # 이미지 저장
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_data = img_byte_arr.getvalue()
    
    return doc_type, content, summary, keywords, structured_data, img_data, embedding

# 벡터 유사도 검색
def search_by_similarity(query, embedding_model, session):
    # 쿼리 임베딩 생성
    query_embedding = create_embedding(query, embedding_model)
    
    # 모든 문서의 임베딩 가져오기
    all_docs = session.exec(select(Document)).all()
    
    similarities = []
    for doc in all_docs:
        if doc.embedding:
            doc_embedding = json.loads(doc.embedding)
            similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            similarities.append((doc, similarity))
    
    # 유사도 순으로 정렬
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, sim in similarities[:10] if sim > 0.5]

# 결과 목록 출력
def print_result_list(results):
    for doc in results:
        with st.expander(f"{doc.filename} - {doc.upload_date.strftime('%Y-%m-%d %H:%M')}"):
            col1, col2 = st.columns(2)
            with col1:
                img = Image.open(io.BytesIO(doc.image_data))
                st.image(img, use_container_width=True)
            with col2:
                st.write(f"**문서 유형:** {doc.doc_type}")
                st.write(f"**요약:** {doc.summary}")
                st.write(f"**키워드:** {doc.keywords}")
                
                if doc.structured_data:
                    structured = json.loads(doc.structured_data)
                    if structured:
                        st.write("**추출된 정보:**")
                        for k, v in structured.items():
                            st.write(f"- {k}: {v}")
                
                # 다운로드
                b64 = base64.b64encode(doc.image_data).decode()
                href = f'<a href="data:image/png;base64,{b64}" download="{doc.filename}">다운로드</a>'
                st.markdown(href, unsafe_allow_html=True)

# Streamlit UI
st.title("AI 아카이브 시스템")

# 모델 로드
with st.spinner("AI 모델 로딩 중..."):
    models = load_models()

# 탭 생성
tab1, tab2, tab3 = st.tabs(["문서 업로드", "문서 검색", "문서 목록"])

if 'processed_file' not in st.session_state:
    st.session_state.processed_file = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'doc_results' not in st.session_state:
    st.session_state.doc_results = None

# 문서 업로드 탭
with tab1:
    uploaded_file = st.file_uploader("문서를 업로드하세요", type=['png', 'jpg', 'jpeg'])

    # 새 파일인지 확인
    if uploaded_file is not None and uploaded_file != st.session_state.processed_file:
        st.session_state.processed_file = uploaded_file
        st.session_state.processing_complete = False
    
    # 처리되지 않은 파일만 처리
    if uploaded_file is not None and not st.session_state.processing_complete:
        with st.spinner("문서 처리 중..."):
            doc_type, content, summary, keywords, structured_data, img_data, embedding = process_document(
                uploaded_file, models
            )
            st.session_state.processing_complete = True
            # 결과를 세션에 저장
            st.session_state.doc_results = {
                'doc_type': doc_type,
                'content': content,
                'summary': summary,
                'keywords': keywords,
                'structured_data': structured_data,
                'img_data': img_data,
                'embedding': embedding
            }
    
    # 처리 완료된 결과 표시
    if uploaded_file is not None and st.session_state.doc_results is not None:
        results = st.session_state.doc_results
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="업로드된 문서", use_container_width=True)
        
        with col2:
            st.write(f"**문서 유형:** {results['doc_type']}")
            st.write(f"**요약:** {results['summary']}")
            st.write(f"**키워드:** {results['keywords']}")
            
            if results['structured_data']:
                st.write("**추출된 정보:**")
                for key, value in results['structured_data'].items():
                    st.write(f"- {key}: {value}")
            
            if st.button("저장"):
                with Session(engine) as session:
                    doc = Document(
                        filename=uploaded_file.name,
                        doc_type=results['doc_type'],
                        content=results['content'],
                        summary=results['summary'],
                        keywords=results['keywords'],
                        structured_data=json.dumps(results['structured_data'], ensure_ascii=False),
                        image_data=results['img_data'],
                        embedding=json.dumps(results['embedding'])
                    )
                    session.add(doc)
                    session.commit()
                st.success("문서가 저장되었습니다!")
                st.session_state.processed_file = None
                st.session_state.processing_complete = False
                st.session_state.doc_results = None

# 문서 검색 탭
with tab2:
    search_query = st.text_input("검색어를 입력하세요 (예: 커피 영수증)")
    search_method = st.radio("검색 방법", ["벡터 유사도 검색", "키워드 검색"])
    
    if st.button("검색", key='search_button'):
        with Session(engine) as session:
            if search_method == "벡터 유사도 검색":
                results = search_by_similarity(
                    search_query, 
                    models[9],  # embedding_model
                    session
                )
            else:
                # 키워드 검색
                statement = select(Document).where(
                    Document.keywords.contains(search_query) | 
                    Document.summary.contains(search_query) |
                    Document.doc_type.contains(search_query)
                )
                results = session.exec(statement).all()
            
            if results:
                print_result_list(results)
            else:
                st.info("검색 결과가 없습니다.")

# 문서 목록 탭
with tab3:
    with Session(engine) as session:
        statement = select(Document)
        results = session.exec(statement).all()
        if results:
            print_result_list(results)
    