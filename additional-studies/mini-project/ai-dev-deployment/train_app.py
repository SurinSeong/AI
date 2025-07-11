from model_train import train_model
from model_regist_loading import load_and_predict, register_model

if __name__ == "__main__":
  # 학습 및 로깅
  model_info = train_model()

  # 모델 등록
  register_model(model_info.model_uri, "mnist-classifier")

  # 추론
  predictions = load_and_predict()
  print(predictions)