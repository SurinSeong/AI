# MLOps 실습 스켈레톤 프로젝트 실행 가이드

## 프로젝트 구조
```
ai-dev-deployment/
├── bentoml/                      # BentoML 서비스 폴더
│   ├── app.py                    # BentoML 서비스 시작점
│   ├── Dockerfile                # BentoML 서비스 도커라이징
│   ├── model_regist_loading.py   # 모델 레지스트리와 로딩
│   ├── model_train.py            # 모델 추적
│   └── requirements.txt          # BentoML 서비스 의존성 목록
├── config/                       # 설정 파일 폴더
│   ├── grafana/                  # Grafana 서비스 설정 폴더 
│   │   └── dashboards          
│   │       └── dashboards.yml    # 대시보드 설정 파일
│   │   └── dashboards_json   
│   │       └── ai_dashboard.json # 대시보드 설정 파일
│   └── prometheus/               # Prometheus 서비스 설정 폴더
│       └── prometheus.yml        # metric 설정 파일
├── data/                         # DVC 실습 데이터 폴더
│   └── large_dataset.csv         # 대용량 데이터셋 추적을 위한 파일
├── metrics/                      # 모델 메트릭 저장 폴더
│   └── train_metrics.json
├── models/                       # DVC 실습 모델 폴더
│   └── pytorch_model.pth         # PyTorch 모델을 DVC로 추적하기 위한 파일
├── .env                          # AWS CLI 설정 파일
├── bentofile.yaml                # PyTorch를 위한 BentoML 서비스 예제 설정 파일
├── deploy_to_ec2.py              # EC2 배포 스크립트
├── deploy-config.yaml            # EC2 배포 설정 파일
├── docker-compose.yml            # 전체 서비스 도커라이징
├── dvc.yaml                      # DVC 실습 예제 파이프라인 설정 파일
├── model_regist_loading.py       # 모델 레지스트리와 로딩 예제 파일
├── model_train.py                # 모델 추적 예제 파일
├── params.yaml                   # DVC 실습 예제 파라미터
├── requirements.txt              # 통합 AI 서비스 의존성 목록
├── sample-predict-data.txt       # BentoML 예제에서 [POST]/predict 호출 시 Request Body로 사용되는 784개 벡터 데이터
├── service.py                    # PyTorch를 위한 BentoML 서비스 예제 파일
├── train_app.py                  # 모델 추적 예제 실행 파일
└── train_pytorch.py              # PyTorch 모델 버전 관리 실습 파일

```

## 빠른 시작
```bash
# 수동 실행
pip install -r requirements.txt

# 모델 추적
python train_app.py

# 모델 버전 관리
python train_pytorch.py

# 도커 컨테이너 실행
docker-compose up -d
```
