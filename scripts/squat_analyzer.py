import cv2
import mediapipe as mp
import numpy as np

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
    """
    어깨 visibility로 정면/측면 자동 감지
    """
    left_shoulder  = landmarks[11]
    right_shoulder = landmarks[12]

    # x좌표 차이로 판단
    x_diff = abs(left_shoulder.x - right_shoulder.x)

    if x_diff > 0.15:
        return "front"
    else:
        return "side"
    
def analyze_front(landmarks, frame):
    """
    정면 분석 - 좌우 비대칭 감지
    """

    # 양쪽 무릎 각도
    l_hip   = [landmarks[23].x, landmarks[23].y]
    l_knee  = [landmarks[25].x, landmarks[25].y]
    l_ankle = [landmarks[27].x, landmarks[27].y]

    r_hip   = [landmarks[24].x, landmarks[24].y]
    r_knee  = [landmarks[26].x, landmarks[26].y]
    r_ankle = [landmarks[28].x, landmarks[28].y]

    l_angle = calculate_angle(l_hip, l_knee, l_ankle)
    r_angle = calculate_angle(r_hip, r_knee, r_ankle)

    # 좌우 차이
    diff = abs(l_angle - r_angle)
    # 화면 표시
    cv2.putText(frame, f'L Knee: {l_angle}',
                (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)
    cv2.putText(frame, f'R Knee: {r_angle}',
                (50, 110), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)
    
    # 비대칭 경고
    if diff > 15:
        cv2.putText(frame, f'WARNING: Asymmetry {diff}',
                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
    return frame
    
def analyze_side(landmarks, frame):
    """
    측면 분석 - 무릎 굴곡각도 + 체간 앞쏠림
    """
    # 무릎 각도
    hip   = [landmarks[24].x, landmarks[24].y]
    knee  = [landmarks[26].x, landmarks[26].y]
    ankle = [landmarks[28].x, landmarks[28].y]

    knee_angle = calculate_angle(hip, knee, ankle)

    # 체간 앞쏠림 (어깨- 엉덩이 - 무릎 각도)
    shoulder = [landmarks[12].x, landmarks[12].y]
    trunk_angle = calculate_angle(shoulder, hip, knee)

    # 화면 표시
    cv2.putText(frame, f'Knee: {knee_angle}',
                (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)
    cv2.putText(frame, f'Trunk: {trunk_angle}',
                (50, 110), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)
    
    # 체간 과도한 압쏠림 경고
    if trunk_angle < 50:
        cv2.putText(frame, 'WARNING: Trunk Forward',
                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
    
    return frame

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

            # 정면/측면 자동 감지
            view = detect_view(landmarks)

            # 모드 표시
            cv2.putText(frame, f'Mode: {view}',
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 0), 2)

            # 모드에 따라 분석
            if view == "front":
                frame = analyze_front(landmarks, frame)
            else:
                frame = analyze_side(landmarks, frame)

            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow('Squat Analyzer', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()