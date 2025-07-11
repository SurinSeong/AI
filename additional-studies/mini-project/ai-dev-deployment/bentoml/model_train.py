import os
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 추적 URI 설정
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:8080')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("pytorch-experiment")

class ConvNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    self.dropout1 = nn.Dropout(0.25)
    self.dropout2 = nn.Dropout(0.5)
    self.fc1 = nn.Linear(9216, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = torch.relu(x)
    x = self.conv2(x)
    x = torch.relu(x)
    x = torch.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = torch.relu(x)
    x = self.dropout2(x)
    x = self.fc2(x)
    return torch.log_softmax(x, dim=1)
  
def train_model():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # 데이터 준비
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
  ])

  train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

  # 모델 초기화
  model = ConvNet().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  criterion = nn.NLLLoss()

  # MLflow 실험 추적
  with mlflow.start_run() as run:
    # 하이퍼파라미터 로깅
    params = {
      "epochs": 5,
      "batch_size": 64,
      "learning_rate": 0.001,
      "optimizer": "Adam",
      "device": str(device)
    }
    mlflow.log_params(params)

    # 학습 루프
    for epoch in range(5):
      model.train()
      epoch_loss = 0

      for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # 100 배치마다 메트릭 로깅
        if batch_idx % 100 == 0:
          step = epoch * len(train_loader) + batch_idx
          mlflow.log_metric("batch_loss", loss.item(), step=step)

      # 에폭 메트릭 로깅
      avg_loss = epoch_loss / len(train_loader)
      mlflow.log_metric("epoch_loss", avg_loss, step=epoch)
      print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")

    # 시그니처와 함께 모델 로깅
    from mlflow.models import infer_signature
    sample_input = torch.randn(1, 1, 28, 28).to(device)

    with torch.no_grad():
      sample_output = model(sample_input)

    signature = infer_signature(
      sample_input.cpu().numpy(),
      sample_output.cpu().detach().numpy()
    )

    # 학습된 모델 로깅
    model_info = mlflow.pytorch.log_model(
      pytorch_model=model,
      artifact_path="model",
      signature=signature,
      input_example=sample_input.cpu().numpy(),
      registered_model_name="mnist-classifier"
    )

    print(f"모델이 로깅됨: {model_info.model_uri}")
    return model_info
