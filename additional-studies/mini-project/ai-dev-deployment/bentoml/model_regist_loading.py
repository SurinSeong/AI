import mlflow
from mlflow.tracking import MlflowClient

# 프로그래밍 방식으로 모델 등록
def register_model(model_uri, model_name):
  client = MlflowClient()

  # 모델 버전 등록
  model_version = client.create_model_version(
    name=model_name,
    source=model_uri,
    description="PyTorch로 학습한 MNIST CNN 분류기"
  )

  # 프로덕션 별칭 설정
  client.set_registered_model_alias(
    name=model_name,
    alias="production",
    version=model_version.version
  )

  print(f"모델 등록됨: {model_name}, 버전: {model_version.version}")
  return model_version

# 추론을 위한 모델 로딩
def load_and_predict():
  # 별칭으로 모델 로드
  model_uri = "models:/mnist-classifier@production"
  loaded_model = mlflow.pyfunc.load_model(model_uri)

  # 테스트 데이터 준비
  import numpy as np
  test_data = np.random.randn(5, 1, 28, 28).astype(np.float32)

  # 예측 수행
  predictions = loaded_model.predict(test_data)
  return predictions
