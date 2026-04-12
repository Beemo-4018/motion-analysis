import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

EMBED_DIM = 62

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

# 모델 로드
model = PoseFormerClassifier()
model.load_state_dict(torch.load('models/poseformer_classifier.pth'))
model.eval()
print('모델 로드 완료')

LABELS = {0: 'standing', 1: 'descending', 2: 'bottom', 3: 'ascending'}
COLORS = {
    0: (0, 255, 0),
    1: (0, 255, 255),
    2: (0, 165, 255),
    3: (255, 165, 0)
}

SEQ_LEN      = 10
frame_buffer = deque(maxlen=SEQ_LEN)
prev_keypoints = None

def extract_keypoints(landmarks):
    selected = [0, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    keypoints = []
    for idx in selected:
        lm = landmarks[idx]
        keypoints.append(lm.x)
        keypoints.append(lm.y)

    def angle(a, b, c):
        a = np.array([landmarks[a].x, landmarks[a].y])
        b = np.array([landmarks[b].x, landmarks[b].y])
        c = np.array([landmarks[c].x, landmarks[c].y])
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) / 180.0

    l_knee = angle(23, 25, 27)
    r_knee = angle(24, 26, 28)
    l_hip  = angle(11, 23, 25)
    r_hip  = angle(12, 24, 26)
    trunk  = angle(0,  11, 23)

    keypoints.extend([l_knee, r_knee, l_hip, r_hip, trunk])
    return np.array(keypoints, dtype=np.float32)

cap = cv2.VideoCapture(0)

with mp_pose.Pose() as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            landmarks  = result.pose_landmarks.landmark
            keypoints  = extract_keypoints(landmarks)

            # 변화율 계산
            if prev_keypoints is not None:
                delta = keypoints - prev_keypoints
            else:
                delta = np.zeros_like(keypoints)

            kp_with_delta = np.concatenate([keypoints, delta])
            prev_keypoints = keypoints
            frame_buffer.append(kp_with_delta)

            if len(frame_buffer) == SEQ_LEN:
                x = torch.tensor(
                    np.array(frame_buffer),
                    dtype=torch.float32
                ).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(x)
                    probs   = torch.softmax(outputs, dim=1)
                    pred    = outputs.argmax(dim=1).item()
                    conf    = probs[0][pred].item()

                label = LABELS[pred]
                color = COLORS[pred]

                cv2.putText(frame, f'Pose: {label}',
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, color, 3)
                cv2.putText(frame, f'Confidence: {round(conf * 100, 1)}%',
                            (50, 90), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, color, 2)

                bar_w = int(conf * 300)
                cv2.rectangle(frame, (50, 115), (50 + bar_w, 140), color, -1)
                cv2.rectangle(frame, (50, 115), (350, 140), (255,255,255), 1)

                y = 170
                for i, (lbl, clr) in enumerate(zip(LABELS.values(), COLORS.values())):
                    p = round(probs[0][i].item() * 100, 1)
                    cv2.putText(frame, f'{lbl}: {p}%',
                                (50, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, clr, 2)
                    y += 25
            else:
                cv2.putText(frame, f'Collecting: {len(frame_buffer)}/{SEQ_LEN}',
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 0), 2)

            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow('PoseFormer Classifier', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()