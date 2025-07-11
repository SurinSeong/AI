from fastapi import FastAPI
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
import time
import GPUtil
from model_train import train_model
from model_regist_loading import load_and_predict, register_model
import bentoml 
from bentoml.exceptions import BentoMLException
import torch
from typing import List

@asynccontextmanager
async def lifespan(app: FastAPI):
  # 앱 시작 시 실행(startup. app.on_event("start") 을 대신함)
  yield
  # 앱 종료 시 실행(필요 시 정리 작업 추가 가능)


#Prometheus instrumentator 초기화
instrumentator = Instrumentator(
  should_group_status_codes=False,
  should_ignore_untemplated=True,
  excluded_handlers=["/metrics", "/health"],
)

#사용자 정의 메트릭
MODEL_TRAINING_LATENCY = Histogram(
  'model_training_latency_seconds',
  '모델 학습에 소요된 시간',
  ['model_name', 'model_version', 'batch_size']
)

MODEL_PREDICTIONS_TOTAL = Counter(
  'model_predictions_total',
  '모델 예측의 총 개수',
  ['model_name', 'status']
)

GPU_UTILIZATION = Gauge(
  'gpu_utilization_percent',
  'GPU 사용률 백분율',
  ['gpu_id', 'gpu_name']
)

def update_gpu_metrics():
  """GPU 메트릭 업데이트"""
  try:
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
      GPU_UTILIZATION.labels(
        gpu_id = gpu.id,
        gpu_name=gpu.name
      ).set(gpu.load * 100)
  except:
    pass
  
app = FastAPI(title="AI Service API", lifespan=lifespan)
instrumentator.instrument(app).expose(app)

@bentoml.service(
  image=bentoml.images.Image(python_version="3.10"),
  resources={"gpu": 1, "memory": "4Gi"}
)
# FastAPI 앱 마운트
@bentoml.asgi_app(app, path="/api/v1")

class PyTorchModelService:
  def __init__(self) -> None:
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model_name = "model.pt"
    self.model = self.load_model()
    self.model.eval()

  def load_model(self):
    #실제 로드할 모델을 입력
    model = torch.jit.load(self.model_name, map_location=self.device)
    return model
  
  @bentoml.api(batchable=True)
  def predict(self, input_data: List[List[float]]) -> List[List[float]]:
    """배치 지원이 있는 단일 예측 엔드포인트"""
    tensor_input = torch.tensor(input_data, dtype=torch.float32).to(self.device)
    with torch.no_grad():
      try:
        predictions = self.model(tensor_input)
        MODEL_PREDICTIONS_TOTAL.labels(
          model_name=self.model_name,
          status="success"
        ).inc()
        result = predictions.cpu().numpy().tolist()
      except Exception as e:
        MODEL_PREDICTIONS_TOTAL.labels(
          model_name=self.model_name,
          status="error"
        ).inc()
        raise BentoMLException(str(e))

    update_gpu_metrics()
    return result

@app.post("/train")
async def predict():
  start_time = time.time()

  # 모델 학습 로직
  model_info = train_model()
  register_model(model_info.model_uri, "mnist-classifier")

  model_name = model_info.registered_model_name
  predictions = load_and_predict()

  result = {"prediction": predictions}
  
  # 학습 지연 시간 기록
  inference_time = time.time() - start_time
  MODEL_TRAINING_LATENCY.labels(
    model_name=model_name,
    model_version="v1.0",
    batch_size=str([1])
  ).observe(inference_time)
  
  update_gpu_metrics()
  return result
  

