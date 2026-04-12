import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 모델 구조 (train_classifier.py랑 동일해야 함)
class PoseClassifier(nn.Module):
    def __init__(self, embed_dim=66, num_classes=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scale   = self.embed_dim ** 0.5
        scores  = torch.matmul(Q, K.transpose(-2, -1)) / scale
        weights = torch.softmax(scores, dim=-1)
        out     = torch.matmul(weights, V)
        out     = out.squeeze(1)
        out     = self.classifier(out)
        return out

# 모델 로드
model = PoseClassifier()
model.load_state_dict(torch.load('models/pose_classifier.pth'))
model.eval()
print('모델 로드 완료')

LABELS = {0: 'standing', 1: 'bottom'}
COLORS = {0: (0, 255, 0), 1: (0, 165, 255)}

def extract_keypoints(landmarks):
    keypoints = []
    for lm in landmarks:
        keypoints.append(lm.x)
        keypoints.append(lm.y)
    return np.array(keypoints)

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
            keypoints = extract_keypoints(landmarks)

            # 모델 추론
            x = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                outputs = model(x)
                probs   = torch.softmax(outputs, dim=1)
                pred    = outputs.argmax(dim=1).item()
                conf    = probs[0][pred].item()

            label = LABELS[pred]
            color = COLORS[pred]

            # 화면 표시
            cv2.putText(frame, f'Pose: {label}',
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, color, 3)
            cv2.putText(frame, f'Confidence: {round(conf * 100, 1)}%',
                        (50, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)

            # Confidence 바
            bar_w = int(conf * 300)
            cv2.rectangle(frame, (50, 120), (50 + bar_w, 145), color, -1)
            cv2.rectangle(frame, (50, 120), (350, 145), (255, 255, 255), 1)

            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow('Pose Classifier', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()