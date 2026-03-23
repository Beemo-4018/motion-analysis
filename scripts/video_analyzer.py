import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return round(angle, 1)

def detect_view(landmarks):
    left_shoulder  = landmarks[11]
    right_shoulder = landmarks[12]
    x_diff = abs(left_shoulder.x - right_shoulder.x)
    if x_diff > 0.08:
        return "front"
    else:
        return "side"

def analyze_front(landmarks, frame):
    l_hip   = [landmarks[23].x, landmarks[23].y]
    l_knee  = [landmarks[25].x, landmarks[25].y]
    l_ankle = [landmarks[27].x, landmarks[27].y]
    r_hip   = [landmarks[24].x, landmarks[24].y]
    r_knee  = [landmarks[26].x, landmarks[26].y]
    r_ankle = [landmarks[28].x, landmarks[28].y]
    l_angle = calculate_angle(l_hip, l_knee, l_ankle)
    r_angle = calculate_angle(r_hip, r_knee, r_ankle)
    diff = abs(l_angle - r_angle)
    cv2.putText(frame, f'L Knee: {l_angle}',
                (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)
    cv2.putText(frame, f'R Knee: {r_angle}',
                (50, 110), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)
    if diff > 15:
        cv2.putText(frame, f'WARNING: Asymmetry {diff}',
                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
    return frame

def analyze_side(landmarks, frame):
    hip      = [landmarks[24].x, landmarks[24].y]
    knee     = [landmarks[26].x, landmarks[26].y]
    ankle    = [landmarks[28].x, landmarks[28].y]
    shoulder = [landmarks[12].x, landmarks[12].y]
    knee_angle  = calculate_angle(hip, knee, ankle)
    trunk_angle = calculate_angle(shoulder, hip, knee)
    cv2.putText(frame, f'Knee: {knee_angle}',
                (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)
    cv2.putText(frame, f'Trunk: {trunk_angle}',
                (50, 110), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)
    if trunk_angle < 50:
        cv2.putText(frame, 'WARNING: Trunk Forward',
                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
    return frame

# 영상 파일 경로
VIDEO_PATH = 'data/raw/IMG_6693.MOV'
SAVE_DIR   = 'results'
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0

with mp_pose.Pose() as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            view = detect_view(landmarks)

            cv2.putText(frame, f'Mode: {view}',
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 0), 2)

            if view == "front":
                frame = analyze_front(landmarks, frame)
            else:
                frame = analyze_side(landmarks, frame)

            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # 30프레임마다 자동 저장
        if frame_count % 30 == 0:
            save_path = f'{SAVE_DIR}/frame_{frame_count}.png'
            cv2.imwrite(save_path, frame)
            print(f'저장됨: {save_path}')

        cv2.imshow('Video Analyzer', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # s 누르면 즉시 저장
        elif key == ord('s'):
            save_path = f'{SAVE_DIR}/manual_{frame_count}.png'
            cv2.imwrite(save_path, frame)
            print(f'수동 저장됨: {save_path}')

cap.release()
cv2.destroyAllWindows()
print(f'완료! results 폴더 확인해보세요.')