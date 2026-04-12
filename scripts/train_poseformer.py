import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split

EMBED_DIM = 62

class PoseDataset(Dataset):
    def __init__(self, data_dir, seq_len=10):
        self.data   = []
        self.labels = []
        self.seq_len = seq_len

        CLASSES = ['standing', 'descending', 'bottom', 'ascending']

        for label_idx, cls in enumerate(CLASSES):
            files = sorted(os.listdir(f'{data_dir}/{cls}'))
            keypoints_list = []

            for f in files:
                path = f'{data_dir}/{cls}/{f}'
                kp   = np.load(path).astype(np.float32)
                keypoints_list.append(kp)

            for i in range(len(keypoints_list) - seq_len + 1):
                seq   = np.array(keypoints_list[i:i + seq_len], dtype=np.float32)
                delta = np.zeros_like(seq)
                delta[1:] = seq[1:] - seq[:-1]
                seq_with_delta = np.concatenate([seq, delta], axis=-1)
                self.data.append(seq_with_delta)
                self.labels.append(label_idx)

        print(f'총 시퀀스: {len(self.data)}개')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data  = torch.tensor(self.data[idx],   dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return data, label


class SpatialTransformer(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, num_heads=2):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm      = nn.LayerNorm(embed_dim)
        self.ff        = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        x = self.norm(x + self.ff(x))
        return x


class TemporalTransformer(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, num_heads=2):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm      = nn.LayerNorm(embed_dim)
        self.ff        = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        x = self.norm(x + self.ff(x))
        return x


class PoseFormerClassifier(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, seq_len=10, num_classes=4):
        super().__init__()
        self.pos_encoding = nn.Parameter(
            torch.randn(1, seq_len, embed_dim)
        )
        self.spatial  = SpatialTransformer(embed_dim)
        self.temporal = TemporalTransformer(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x + self.pos_encoding
        x = self.spatial(x)
        x = self.temporal(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x


def train():
    dataset    = PoseDataset('data', seq_len=10)
    train_size = int(len(dataset) * 0.8)
    val_size   = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=32)

    model     = PoseFormerClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f'학습: {train_size}개 / 검증: {val_size}개')
    print('학습 시작...\n')

    best_val_acc = 0

    for epoch in range(100):
        model.train()
        train_loss    = 0
        train_correct = 0
        train_total   = 0

        for keypoints, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(keypoints)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            predicted      = outputs.argmax(dim=1)
            train_correct += (predicted == labels).sum().item()
            train_total   += labels.size(0)

        model.eval()
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for keypoints, labels in val_loader:
                outputs   = model(keypoints)
                predicted = outputs.argmax(dim=1)
                val_correct += (predicted == labels).sum().item()
                val_total   += labels.size(0)

        train_acc = train_correct / train_total * 100
        val_acc   = val_correct   / val_total   * 100

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:3d}/100  '
                  f'Train Loss: {train_loss:.4f}  '
                  f'Train Acc: {train_acc:.1f}%  '
                  f'Val Acc: {val_acc:.1f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/poseformer_classifier.pth')

    print(f'\n학습 완료! 최고 Val Acc: {best_val_acc:.1f}%')
    print('모델 저장: models/poseformer_classifier.pth')


if __name__ == "__main__":
    train()