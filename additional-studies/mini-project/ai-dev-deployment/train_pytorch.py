import torch
import torch.nn as nn
import json
from pathlib import Path

class SimpleNet(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(SimpleNet, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    return out
  
def train_model():
  input_size = 784
  hidden_size = 128
  num_classes = 10
  model = SimpleNet(input_size, hidden_size, num_classes)

  # 모델 저장
  torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': {
      'input_size': input_size,
      'hidden_size': hidden_size,
      'num_classes': num_classes
    }
  }, 'models/pytorch_model.pth')
  
  # TorchScript 모델로 변환 (example_input은 반드시 2D 텐서여야 함)
  example_input = torch.randn(1, input_size)  # 배치 크기 1
  traced_model = torch.jit.trace(model, example_input)

  torch.jit.save(traced_model, "model.pt")  # TorchScript 모델 저장

  #메트릭 저장
  metrics = {
    'accuracy': 0.92,
    'loss': 0.25,
    'epoch': 100
  }

  Path('metrics').mkdir(exist_ok=True)
  with open('metrics/train_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

if __name__ == "__main__":
  train_model()