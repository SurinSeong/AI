import re
import torch
import json

# 영수증 구조 추출
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

# 문서 구조 추출
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