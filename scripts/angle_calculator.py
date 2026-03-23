import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """
    세 점으로 관절각도 계산
    a = 고관절, b = 무릎, c = 발목
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # 벡터 계산
    ba = a - b
    bc = c - b

    # 코사인 법칙으로 각도 계산
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    return round(angle, 1)

cap = cv2.VideoCapture(0)

with mp_pose.Pose() as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark

            # 오른쪽 무릎 각도
            hip   = [landmarks[24].x, landmarks[24].y]
            knee  = [landmarks[26].x, landmarks[26].y]
            ankle = [landmarks[28].x, landmarks[28].y]

            angle = calculate_angle(hip, knee, ankle)

            # 화면에 각도 표시
            cv2.putText(frame, f'Knee: {angle}',
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow('Angle Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()