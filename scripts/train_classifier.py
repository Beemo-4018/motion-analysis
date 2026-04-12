import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

# 데이터셋 클래스
class PoseDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        self.labels = []

        # standing 로드 (label 0)
        for f in os.listdir(f'{data_dir}/standing'):
            path = f'{data_dir}/standing/{f}'
            keypoints = np.load(path)
            self.data.append(keypoints)
            self.labels.append(0)

        # bottom 로드 (label 1)
        for f in os.listdir(f'{data_dir}/bottom'):
            path = f'{data_dir}/bottom/{f}'
            keypoints = np.load(path)
            self.data.append(keypoints)
            self.labels.append(1)

        self.data   = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Attention 기반 분류 모델
class PoseClassifier(nn.Module):
    def __init__(self, embed_dim=66, num_classes=2):
        super().__init__()
        self.embed_dim = embed_dim

        # Self-Attention
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # 분류기
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, 66)
        # 단일 프레임이라 unsqueeze로 시퀀스 차원 추가
        x = x.unsqueeze(1)  # (batch, 1, 66)

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scale   = self.embed_dim ** 0.5
        scores  = torch.matmul(Q, K.transpose(-2, -1)) / scale
        weights = torch.softmax(scores, dim=-1)
        out     = torch.matmul(weights, V)

        out = out.squeeze(1)  # (batch, 66)
        out = self.classifier(out)
        return out

# 학습
def train():
    dataset    = PoseDataset('data')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model     = PoseClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f'데이터 총 {len(dataset)}개')
    print('학습 시작...')

    for epoch in range(50):
        total_loss    = 0
        correct       = 0
        total         = 0

        for keypoints, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(keypoints)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted   = outputs.argmax(dim=1)
            correct    += (predicted == labels).sum().item()
            total      += labels.size(0)

        acc = correct / total * 100
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/50  Loss: {total_loss:.4f}  Acc: {acc:.1f}%')

    # 모델 저장
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/pose_classifier.pth')
    print('모델 저장 완료: models/pose_classifier.pth')

if __name__ == "__main__":
    train()