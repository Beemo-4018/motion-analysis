import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Self-Attention 모듈
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scale  = self.embed_dim ** 0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)
        return out, weights

# 설정
SEQ_LEN   = 10   # 몇 프레임 모을지
EMBED_DIM = 66   # 33개 관절 x (x, y)

# 프레임 버퍼
frame_buffer = []
attention    = SelfAttention(EMBED_DIM)

def extract_keypoints(landmarks):
    """MediaPipe 랜드마크 → 66차원 벡터"""
    keypoints = []
    for lm in landmarks:
        keypoints.append(lm.x)
        keypoints.append(lm.y)
    return keypoints  # 66개 값

def get_attention_weights(buffer):
    """프레임 버퍼 → Attention 가중치"""
    x = torch.tensor(buffer, dtype=torch.float32)
    x = x.unsqueeze(0)  # (1, seq_len, 66)
    with torch.no_grad():
        _, weights = attention(x)
    # 각 프레임의 평균 가중치
    avg_weights = weights[0].mean(dim=0).numpy()
    return avg_weights

cap = cv2.VideoCapture(0)

with mp_pose.Pose() as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark

            # 관절 좌표 추출
            keypoints = extract_keypoints(landmarks)
            frame_buffer.append(keypoints)

            # 버퍼가 SEQ_LEN 넘으면 오래된 것 제거
            if len(frame_buffer) > SEQ_LEN:
                frame_buffer.pop(0)

            # SEQ_LEN 프레임 모이면 Attention 계산
            if len(frame_buffer) == SEQ_LEN:
                weights = get_attention_weights(frame_buffer)

                # 가장 중요한 프레임
                most_important = np.argmax(weights)
                max_weight     = weights[most_important]

                # 화면 표시
                cv2.putText(frame, f'Frames: {len(frame_buffer)}/{SEQ_LEN}',
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 0), 2)
                cv2.putText(frame, f'Most important frame: {most_important}',
                            (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Weight: {round(float(max_weight), 3)}',
                            (50, 110), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

                # 가중치 막대 시각화
                for i, w in enumerate(weights):
                    bar_h = int(w * 300)
                    x_pos = 50 + i * 40
                    cv2.rectangle(frame,
                                (x_pos, 300 - bar_h),
                                (x_pos + 30, 300),
                                (0, 165, 255), -1)
                    cv2.putText(frame, str(i),
                                (x_pos + 8, 315),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 1)
            else:
                cv2.putText(frame, f'Collecting: {len(frame_buffer)}/{SEQ_LEN}',
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 0), 2)

            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow('Pose Attention', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()