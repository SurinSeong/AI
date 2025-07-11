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
import cv2
import io
import base64
import numpy as np
from typing import Optional
import json
import re
from sklearn.metrics.pairwise import cosine_similarity

from preprocess_image import *
from analyze_morpheme import *
from get_info_in_just_image import *
from extract_info import *

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
    if any(keyword in predicted_class.lower() for keyword in ['invoice', 'receipt']):
        return "영수증"
    
    # 문서 관련 클래스가 아니면 "일반 사진"으로 반환
    if any(keyword in predicted_class.lower() for keyword in ['image', 'photo', 'landscape', 'nature']):
        return "일반 사진"
    
    return predicted_class


# OCR 성능 향상을 위한 문서 이미지 전처리
def preprocess_image_for_ocr(image, blur_size, block_size, C_value, dilation_iter, to_grayscale):
    """OCR 성능 향상을 위한 이미지 전처리"""
    # PIL을 openCV로 변환
    img_np = np.array(image)  # Pillow → numpy
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 1. 그레이 스케일 변화
    if to_grayscale:
        gray_image = convert_to_grayscale(img_cv)
    else:
        gray_image = img_cv

    # 2. 노이즈 제거
    denoised = remove_noise(gray_image, blur_size)

    # 3. 대비 개선
    if to_grayscale:
        enhanced = improve_contrast(denoised)
    else:
        enhanced = denoised

    # 4. 이진화 (텍스트와 배경 분리)
    if to_grayscale:
        binary = apply_adaptive_binarization(enhanced, block_size, C_value)
    else:
        binary = enhanced

    # 5. 텍스트 영역 강화
    final_image = enhance_text_regions(binary, dilation_iter)

    return final_image

# OCR 수행
def perform_ocr(image, ocr):
    result = ocr.ocr(np.array(image))
    return result

# 텍스트와 위치 정보 추출
def extract_text_and_positions(result):
    text = ""
    boxes = []
    
    if result[0]:
        for line in result[0]:
            text += line[1][0] + " "
            boxes.append(line[0])

    return text.strip(), boxes
    
# 기존 OCR 함수에 전처리 옵션 추가하기
def extract_text_with_layout(ocr, image, blur_size, block_size, C_value, dilation_iter, to_grayscale, use_preprocessing=True):
    """전처리 옵션이 추가된 OCR 함수"""
    # 전처리 적용 여부 확인하기
    if use_preprocessing:
        image = preprocess_image_for_ocr(image, blur_size, block_size, C_value, dilation_iter, to_grayscale)

    # OCR 수행하기
    result = perform_ocr(image, ocr)

    # 텍스트와 위치 정보 추출하기
    text, boxes = extract_text_and_positions(result)
    
    return text, boxes

#---------------------------------------------------------------------------------
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

# 구조화된 데이터에서 키워드 추출하기 함수
def extract_structured_keywords(structured_data):
    structured_data_keywords = []

    if 'store' in structured_data:
        structured_data_keywords.append(structured_data['store'])
    if 'date' in structured_data:
        structured_data_keywords.append(structured_data['date'])

    return structured_data_keywords

# 기존 함수 대체하기
def extract_keywords(text, structured_data=None):
    """개선된 키워드 추출 함수"""
    # 형태소 분석 기반 키워드 추출
    keywords, method = extract_keywords_with_morpheme_analysis(text)
    
    # 구조화된 데이터에서 추가 키워드
    if structured_data:
        keywords.extend(extract_structured_keywords(structured_data))
    
    return ", ".join(list(set(keywords)))

#---------------------------------------------------------------------------
# 사진 구분 및 메타데이터 검색
# 사진 안의 객체 탐지하기
def detect_photo_objects(image):
    """사진 내의 객체 탐지"""
    detected_objects = run_object_detection_model(image)

    # 디버깅
    for obj in detected_objects:
        print(f"Detected {obj['class']} with confidence {obj['confidence']:.2f}")

    return detected_objects

#------------------------------------------------------------
# 문서 처리
def process_document(uploaded_file, models, blur_size, block_size, C_value, dilation_iter, to_grayscale):
    (dit_processor, dit_model, ocr, donut_processor, donut_model, 
     layout_processor, layout_model, sum_tokenizer, sum_model,
     embedding_model) = models
    
    # 이미지를 Pillow로 변환 -> RGB
    image = Image.open(uploaded_file).convert("RGB")
    
    # 1. 문서 유형 분류
    doc_type = classify_document(image, dit_processor, dit_model)
    
    print('\n==========')
    print(f'\n[doc_type]\n{doc_type}')
    
    content, boxes = extract_text_with_layout(image, ocr, blur_size, block_size, C_value, dilation_iter, to_grayscale)
    layoutlm_data = extract_structured_with_layoutlm(image, content, boxes, layout_processor, layout_model, doc_type)
    
    print('\n==========')
    print(f'\n[layoutlm_data]\n{layoutlm_data}')
    
    # 이거 어디에 들어가야 하는지 생각하기
    # 사진 여부 판별하기
    if is_photo(doc_type, content):
        # 메타데이터 추출
        metadata = extract_photo_metadata(image)
        
        # 객체 탐지
        objects = detect_photo_objects(image)

        # 키워드 생성
        keywords = generate_photo_keywords(metadata, objects)

        # 구조화된 데이터로 저장
        structured_data = format_photo_data(metadata, objects)

        # 요약 생성
        summary = create_photo_summary(objects, metadata)

        # 임베딩 생성
        embedding = create_embedding(summary, embedding_model)

        return doc_type, content, summary, keywords, structured_data, img_data, embedding

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

#-------------------------------------------------------------------------------------------------
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

    # 새 파일이면 세션 초기화 하기 + 이전 결과 백업하기
    if uploaded_file is not None and uploaded_file != st.session_state.processed_file:
        # 이전 결과 백업
        if st.session_state.doc_results:
            st.session_state.prev_doc_results = st.session_state.doc_results.copy()

        st.session_state.processed_file = uploaded_file
        st.session_state.processing_complete = False
        st.session_state.ocr_ready = False    # OCR은 버튼 누를 때만 진행한다.

    # 파일이 업로드 되었다면
    if uploaded_file:
        st.subheader("🔧 전처리 설정")

        grayscale_option = st.checkbox("흑백(Grayscale) 처리", value=True)
        blur_size = st.slider("가우시안 블러 커널 크기", 1, 11, 5, step=2)
        block_size = st.slider("Adaptive Threshold blockSize", 3, 25, 11, step=2)
        C_value = st.slider("Threshold 상수 C", 0, 10, 2)
        dilation_iter = st.slider("Dilation 반복 횟수", 0, 3, 1)

        # 전처리 미리보기
        pil_image = Image.open(uploaded_file).convert("RGB")
        final_image = preprocess_image_for_ocr(pil_image, blur_size, block_size, C_value, dilation_iter, grayscale_option)

        if grayscale_option:
            channel_name = "GRAY"
        else:
            channel_name = "COLOR"
        
        st.image(final_image, caption="전처리 결과", channels=channel_name)
        
        if st.button("🔠 OCR 시작"):
            with st.spinner("OCR 분석 중..."):
                doc_type, content, summary, keywords, structured_data, img_data, embedding = process_document(
                    uploaded_file, models, blur_size, block_size, C_value, dilation_iter, grayscale_option
                )
                st.session_state.processing_complete = True
                st.session_state.ocr_ready = True
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
        
    prev_results = st.session_state.get("prev_doc_results")

    # 처리 완료된 결과 표시
    if uploaded_file is not None and st.session_state.doc_results is not None:
        results = st.session_state.doc_results
        

        st.subheader("📄 현재 분석된 문서 결과")
        
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

    # 이전 결과가 있다면 비교하기
    if prev_results:
        st.markdown("---")
        st.subheader("🔁 이전 분석 결과와 비교")

        col1, col2 = st.columns(2)

        with col1:
            st.write("📄 **이전 요약**")
            st.write(prev_results["summary"])
            st.write("🔑 **이전 키워드**")
            st.write(prev_results["keywords"])

            if prev_results.get("structured_data"):
                st.write("📋 **이전 추출 정보**")
                for key, value in prev_results["structured_data"].items():
                    st.write(f"- {key}: {value}")

        with col2:
            st.write("📄 **현재 요약**")
            st.write(results["summary"])
            st.write("🔑 **현재 키워드**")
            st.write(results["keywords"])

            if results.get("structured_data"):
                st.write("📋 **현재 추출 정보**")
                for key, value in results["structured_data"].items():
                    st.write(f"- {key}: {value}")

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
    