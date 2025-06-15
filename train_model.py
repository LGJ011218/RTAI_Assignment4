import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 데이터셋 로딩
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data', train=True, download=True,
                          transform=transforms.ToTensor()),
    batch_size=64, shuffle=True
)

# 모델 학습
model = SimpleMLP()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# 디렉토리 생성
os.makedirs("complete_verifier/models", exist_ok=True)

# PyTorch 모델 저장
torch.save(model.state_dict(), "alpha-beta-CROWN/complete_verifier/models/fashion_mnist_mlp.pth")

# ✅ ONNX로도 저장
model.eval()
dummy_input = torch.randn(1, 1, 28, 28)
torch.onnx.export(
    model,
    dummy_input,
    "alpha-beta-CROWN/complete_verifier/models/fashion_mnist_mlp.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print("✅ PyTorch와 ONNX 모델 저장 완료!")
