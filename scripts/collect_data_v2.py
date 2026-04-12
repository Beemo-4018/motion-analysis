import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

CLASSES = ['standing', 'descending', 'bottom', 'ascending']
for c in CLASSES:
    os.makedirs(f'data/{c}', exist_ok=True)

def extract_keypoints(landmarks):
    # 스쿼트 핵심 관절만 선택
    # 코(0), 어깨(11,12), 엉덩이(23,24), 무릎(25,26), 발목(27,28), 발(29,30,31,32)
    selected = [0, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

    keypoints = []
    for idx in selected:
        lm = landmarks[idx]
        keypoints.append(lm.x)
        keypoints.append(lm.y)

    # 각도 특징 추가
    def angle(a, b, c):
        a = np.array([landmarks[a].x, landmarks[a].y])
        b = np.array([landmarks[b].x, landmarks[b].y])
        c = np.array([landmarks[c].x, landmarks[c].y])
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) / 180.0

    l_knee = angle(23, 25, 27)  # 왼쪽 무릎
    r_knee = angle(24, 26, 28)  # 오른쪽 무릎
    l_hip  = angle(11, 23, 25)  # 왼쪽 고관절
    r_hip  = angle(12, 24, 26)  # 오른쪽 고관절
    trunk  = angle(0,  11, 23)  # 체간 기울기

    keypoints.extend([l_knee, r_knee, l_hip, r_hip, trunk])

    return np.array(keypoints)  # 13관절 x 2 + 각도 5 = 31차원

cap = cv2.VideoCapture(0)
current_label = None
counts = {c: 0 for c in CLASSES}
TARGET = 150

print("키 입력 안내 (OpenCV 창 클릭 후 누르세요)")
print("1 → standing")
print("2 → descending")
print("3 → bottom")
print("4 → ascending")
print("q → 종료")

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

            if current_label and counts[current_label] < TARGET:
                save_path = f'data/{current_label}/{counts[current_label]}.npy'
                np.save(save_path, keypoints)
                counts[current_label] += 1

            if current_label and counts[current_label] >= TARGET:
                print(f'{current_label} 수집 완료!')
                current_label = None

            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        cv2.putText(frame, f'Collecting: {current_label}',
                    (50, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 0), 2)

        y = 70
        colors = [(0,255,0), (0,255,255), (0,165,255), (255,165,0)]
        for i, c in enumerate(CLASSES):
            done = counts[c] >= TARGET
            text = f'{i+1}. {c}: {counts[c]}/{TARGET} {"✓" if done else ""}'
            cv2.putText(frame, text,
                        (50, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, colors[i], 2)
            y += 30

        cv2.imshow('Data Collection v2', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            print('3초 후 standing 수집 시작...')
            cv2.waitKey(3000)
            current_label = 'standing'
            print('standing 수집 시작')
        elif key == ord('2'):
            print('3초 후 descending 수집 시작...')
            cv2.waitKey(3000)
            current_label = 'descending'
            print('descending 수집 시작')
        elif key == ord('3'):
            print('3초 후 bottom 수집 시작...')
            cv2.waitKey(3000)
            current_label = 'bottom'
            print('bottom 수집 시작')
        elif key == ord('4'):
            print('3초 후 ascending 수집 시작...')
            cv2.waitKey(3000)
            current_label = 'ascending'
            print('ascending 수집 시작')

cap.release()
cv2.destroyAllWindows()
print('수집 완료!')
for c in CLASSES:
    print(f'{c}: {counts[c]}개')