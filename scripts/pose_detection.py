import cv2
import mediapipe as mp

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 웹캠 열기
cap = cv2.VideoCapture(0)

with mp_pose.Pose() as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR → RGB 변환
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 관절 추출
        result = pose.process(rgb)

        # 관절 그리기
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow('Pose Detection', frame)

        # q 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()