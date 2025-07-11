import cv2

# 그레이 스케일 변환하기
def convert_to_grayscale(img_cv):
    gray_image = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    return gray_image

# 노이즈 제거하기
def remove_noise(gray_image, blur_size):
    # 가우시안 블러 사용 => 커널의 크기가 클수록 더 부드럽게 된다. 하지만 너무 크면 뭉개짐.
    # 얇은 글자일 경우 -> 작게 / 잉크 번짐과 같은 노이즈가 많을 경우 -> 조금 더 크게
    denoised = cv2.GaussianBlur(gray_image, (blur_size, blur_size), 0)
    return denoised

# 대비 개선하기
def improve_contrast(denoised):
    # 전체적인 대비 향상 -> 조명에 따라 부자연스러울 수 있음.
    # enhanced = cv2.equalizeHist(denoised)
    # 더 좋은 대안 : CLAHE (적응적 히스토그램 평활화)
    # clipLimit가 낮을수록 부드럽고, 높을수록 대비가 강조된다.
    # tileGridSize는 글자 크기 기준 보통 (8, 8), 글자가 작으면 (4, 4)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised) 
    return enhanced

# 이진화
def apply_adaptive_binarization(enhanced, block_size, C_value):
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2. THRESH_BINARY,
        block_size,     # 주변 블록 크기. 글자 크기보다 조금 큰 값이 좋음.
        C_value       # 빼는 상수. 어두운 배경일수록 조금 더 크게 조절 가능.
    )
    return binary

# 텍스트 영역 강화
def enhance_text_regions(binary, dilation_iter):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 텍스트 굵게 해서 OCR이 잘 되도록 할 수 있다.
    # 얇은 글씨 : iterations=1
    # 끊긴 글씨나 번진 잉크는 (3, 3)보다 큰 커널로 iterations=2~3도 실험해볼만하다.
    final_image = cv2.dilate(binary, kernel, iterations=dilation_iter)
    return final_image
