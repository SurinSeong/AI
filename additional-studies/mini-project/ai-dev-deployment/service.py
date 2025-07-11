import bentoml
import torch
from pydantic import BaseModel
from typing import List

@bentoml.service(
  image=bentoml.images.Image(python_version="3.10"),
  resources={"gpu": 1, "memory": "4Gi"}
)

class PyTorchModelService:
  def __init__(self) -> None:
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model = self.load_model()
    self.model.eval()

  def load_model(self):
    model = torch.jit.load("model.pt", map_location=self.device)
    return model
  
  @bentoml.api(batchable=True)
  def predict(self, input_data: List[List[float]]) -> List[List[float]]:
    """배치 지원이 있는 단일 예측 엔드포인트"""
    tensor_input = torch.tensor(input_data, dtype=torch.float32).to(self.device)
    with torch.no_grad():
      predictions = self.model(tensor_input)
    return predictions.cpu().numpy().tolist()
  
  @bentoml.api(batchable=True, batch_dim=0)
  def batch_predict(self, batch_data: List[List[float]]) -> List[List[float]]:
    """전용 배치 예측 엔드포인트"""
    tensor_batch = torch.tensor(batch_data, dtype=torch.float32).to(self.device)
    with torch.no_grad():
      batch_predictions = self.model(tensor_batch)
    return batch_predictions.cpu().numpy().tolist()
  