import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 저장 폴더
os.makedirs('data/standing', exist_ok=True)
os.makedirs('data/bottom', exist_ok=True)

def extract_keypoints(landmarks):
    keypoints = []
    for lm in landmarks:
        keypoints.append(lm.x)
        keypoints.append(lm.y)
    return np.array(keypoints)

cap = cv2.VideoCapture(0)
current_label = None
counts = {'standing': 0, 'bottom': 0}
TARGET = 100  # 각 동작 100프레임

print("s 누르면 standing 수집 시작")
print("b 누르면 bottom 수집 시작")
print("q 누르면 종료")

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

            # 수집 중이면 저장
            if current_label and counts[current_label] < TARGET:
                save_path = f'data/{current_label}/{counts[current_label]}.npy'
                np.save(save_path, keypoints)
                counts[current_label] += 1

            # 목표 달성 시 중지
            if current_label and counts[current_label] >= TARGET:
                print(f'{current_label} 수집 완료!')
                current_label = None

            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # 상태 표시
        cv2.putText(frame, f'Collecting: {current_label}',
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 0), 2)
        cv2.putText(frame, f'standing: {counts["standing"]}/{TARGET}',
                    (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'bottom: {counts["bottom"]}/{TARGET}',
                    (50, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 165, 255), 2)

        cv2.imshow('Data Collection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print('3초 후 standing 수집 시작...')
            cv2.waitKey(3000)
            current_label = 'standing'
            print('standing 수집 시작')
        elif key == ord('b'):
            print('3초 후 bottom 수집 시작...')
            cv2.waitKey(3000)
            current_label = 'bottom'
            print('bottom 수집 시작')

cap.release()
cv2.destroyAllWindows()
print(f'최종 수집: standing {counts["standing"]}개, bottom {counts["bottom"]}개')