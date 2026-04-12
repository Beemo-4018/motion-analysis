import torch
import torch.nn as nn
import numpy as np

# 간단한 Self-Attention 구현
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # Query, Key, Value 변환 레이어
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        Q = self.W_q(x)  # Query
        K = self.W_k(x)  # Key
        V = self.W_v(x)  # Value

        # Attention Score 계산
        # Q와 K가 얼마나 비슷한지
        scale = self.embed_dim ** 0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        # Softmax로 가중치 변환 (합이 1이 되도록)
        weights = torch.softmax(scores, dim=-1)

        # Value에 가중치 적용
        out = torch.matmul(weights, V)

        return out, weights
    
# 테스트
if __name__ == "__main__":

    # 스쿼트 10프레임, 각 프레임마다 관절 66개 (33개 x 2D)
    batch_size = 1
    seq_len    = 10   # 10프레임
    embed_dim  = 66   # 33개 관절 x (x, y)

    # 랜덤 관절 좌표 생성 (실제로는 MediaPipe 좌표)
    x = torch.randn(batch_size, seq_len, embed_dim)
    print(f'입력 shape: {x.shape}')

    # Self-Attention 실행
    attention = SelfAttention(embed_dim)
    out, weights = attention(x)

    print(f'출력 shape: {out.shape}')
    print(f'Attention 가중치 shape: {weights.shape}')
    print(f'Attention 가중치 합계 (행마다 1이어야 함):')
    print(weights[0][0])  # 첫 번째 프레임의 가중치