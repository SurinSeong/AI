# NVIDIA CUDA 베이스 이미지 사용
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Python 3.10 및 시스템 종속성 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 생성
WORKDIR /app

# 추론 API 제공할 모델 복사
COPY ./model.pt .
# ML 종속성 설치
COPY ./requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# RTX 5090 최적화 설정
ENV TORCH_CUDNN_V8_API_ENABLED=1
ENV CUDA_VISIBLE_DEVICES=0

# 포트 노출
EXPOSE 8000

# 애플리케이션 실행
CMD ["bentoml", "serve", "app:PyTorchModelService", "--port", "8000"]