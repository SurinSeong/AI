from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from konlpy.tag import Komoran

komoran = Komoran()

# 형태소 분석 및 품사 태깅하기
def morpheme_analyze(text):
    pos_tagged = komoran.pos(text)
    return pos_tagged

# 명사만 추출하는 함수
def extract_nouns_from_pos(pos_tagged):
    nouns = [word for word, tag in pos_tagged if tag == "Noun" and len(word) > 1]
    return list(set(nouns))

# 불용어 필터링 함수
def filter_stopwords(nouns, stopwords=None):
    if stopwords is None:
        stopwords = ['은', '는', '이', '가', '을', '를', '의', '에', '와', '과', '에서', '으로']

    filtered = [word for word in nouns if word not in stopwords]
    return filtered

# 복합 명사 생성
def create_compound_nouns(pos_tagged):

    compound_nouns = []
    temp = []

    # 태깅된 단어들을 순회한다.
    for word, tag in pos_tagged:
        if tag in ["NNG", "NNP"]:    # 일반 명사, 고유 명사
            temp.append(word)

        # 명사 태그가 없다면
        else:
            if len(temp) > 1:
                compound_nouns.append(''.join(temp))
            temp = []

    if len(temp) > 1:
        compound_nouns.append(''.join(temp))

    return compound_nouns

# TF-IDF 점수를 통한 키워드 중요도 계산하기
def calculate_tfidf_scores(current_nouns, type_grouped_nouns, threshold=3):
    """
    current_nouns: 현재 문서의 명사 리스트
    type_grouped_nouns: 동일 유형의 전체 문서들의 명사 리스트
    threshold: TF-IDF를 적용할 최소 문서의 수
    """
    if len(type_grouped_nouns) >= threshold:
        # TF-IDF 적용하기
        all_docs = type_grouped_nouns + [current_nouns]
        docs = [' '.join(doc) for doc in all_docs]

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(docs)

        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[-1]    # 마지막이 현재의 문서이기 때문에

        word_scores = dict(zip(feature_names, scores))
        method = "tfidf"

    else:
        # 단어 빈도 적용
        word_scores = dict(Counter(current_nouns))
        method = "term-frequency"
    
    return word_scores, method

# 점수 기준 상위 키워드 선택
def select_top_keywords(word_scores, top_k):
    # 점수 기준 내림차순으로 정렬
    sorted_keywords = sorted(word_scores.items(), key=lambda x:x[1], reverse=True)

    # 상위 top_k만 선택하기
    top_keywords = []
    for word, score in sorted_keywords[:top_k]:
        top_keywords.append(word)
    
    return top_keywords

# 형태소 분석을 통한 키워드 추출
def extract_keywords_with_morpheme_analysis(text, top_k=15):
    """형태소 분석을 통한 키워드 추출"""
    # 1. 형태소 분석 및 품사 태깅
    pos_tagged = morpheme_analyze(text)

    # 2. 명사만 추출 (일반 명사, 고유 명사)
    nouns = extract_nouns_from_pos(pos_tagged)

    # 불용어 필터링
    filtered_nouns = filter_stopwords(nouns)

    # 3. 복합 명사 생성
    compound_nouns = create_compound_nouns(pos_tagged)

    # 복합 명사 + 일반 명사 + 고유 명사
    filtered_nouns.extend(compound_nouns)

    # 4. TF-IDF 점수 계산
    # 우선 같은 유형의 문서가 없다고 가정한다.
    word_scores, method = calculate_tfidf_scores(filtered_nouns, [])

    # 5. 점수 기준 상위 키워드 선택
    top_keywords = select_top_keywords(word_scores, top_k)

    return top_keywords, method
