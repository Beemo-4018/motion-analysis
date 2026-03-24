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

# 스쿼트 상태 변수
squat_state = "standing"
squat_count = 0

def detect_view(landmarks):
    """
    어깨 visibility로 정면/측면 자동 감지
    """
    left_shoulder  = landmarks[11]
    right_shoulder = landmarks[12]

    # x좌표 차이로 판단
    x_diff = abs(left_shoulder.x - right_shoulder.x)

    if x_diff > 0.08:
        return "front"
    else:
        return "side"
    
def analyze_front(landmarks, frame):
    global squat_state, squat_count

    l_hip   = [landmarks[23].x, landmarks[23].y]
    l_knee  = [landmarks[25].x, landmarks[25].y]
    l_ankle = [landmarks[27].x, landmarks[27].y]
    r_hip   = [landmarks[24].x, landmarks[24].y]
    r_knee  = [landmarks[26].x, landmarks[26].y]
    r_ankle = [landmarks[28].x, landmarks[28].y]

    l_angle = calculate_angle(l_hip, l_knee, l_ankle)
    r_angle = calculate_angle(r_hip, r_knee, r_ankle)
    avg_angle = (l_angle + r_angle) / 2
    diff = abs(l_angle - r_angle)

    # 동작 단계 인식 (평균 각도 기준)
    if avg_angle > 160:
        if squat_state == "ascending":
            squat_count += 1
        squat_state = "standing"
    elif avg_angle <= 160 and avg_angle > 120:
        if squat_state == "standing":
            squat_state = "descending"
        elif squat_state == "bottom":
            squat_state = "ascending"
    elif avg_angle <= 120:
        squat_state = "bottom"

    state_color = {
        "standing":   (0, 255, 0),
        "descending": (0, 255, 255),
        "bottom":     (0, 165, 255),
        "ascending":  (255, 165, 0)
    }
    color = state_color[squat_state]

    cv2.putText(frame, f'L Knee: {l_angle}',
                (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2)
    cv2.putText(frame, f'R Knee: {r_angle}',
                (50, 110), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2)
    cv2.putText(frame, f'State: {squat_state}',
                (50, 140), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2)
    cv2.putText(frame, f'Count: {squat_count}',
                (50, 170), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2)

    if diff > 15:
        cv2.putText(frame, f'WARNING: Asymmetry {round(diff, 1)}',
                    (50, 210), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

    return frame
    
def analyze_side(landmarks, frame):
    global squat_state, squat_count

    hip   = [landmarks[24].x, landmarks[24].y]
    knee  = [landmarks[26].x, landmarks[26].y]
    ankle = [landmarks[28].x, landmarks[28].y]
    shoulder = [landmarks[12].x, landmarks[12].y]

    knee_angle  = calculate_angle(hip, knee, ankle)
    trunk_angle = calculate_angle(shoulder, hip, knee)

    # 동작 단계 인식
    if knee_angle > 160:
        if squat_state == "ascending":
            squat_count += 1
        squat_state = "standing"
    elif knee_angle <= 160 and knee_angle > 120:
        if squat_state == "standing":
            squat_state = "descending"
        elif squat_state == "bottom":
            squat_state = "ascending"
    elif knee_angle <= 120:
        squat_state = "bottom"

    # 단계별 색상
    state_color = {
        "standing":   (0, 255, 0),
        "descending": (0, 255, 255),
        "bottom":     (0, 165, 255),
        "ascending":  (255, 165, 0)
    }
    color = state_color[squat_state]

    # 화면 표시
    cv2.putText(frame, f'Knee: {knee_angle}',
                (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2)
    cv2.putText(frame, f'Trunk: {trunk_angle}',
                (50, 110), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)
    cv2.putText(frame, f'State: {squat_state}',
                (50, 140), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2)
    cv2.putText(frame, f'Count: {squat_count}',
                (50, 170), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2)

    if trunk_angle < 50:
        cv2.putText(frame, 'WARNING: Trunk Forward',
                    (50, 210), cv2.FONT_HERSHEY_SIMPLEX,
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

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # s 누르면 즉시 저장
        elif key == ord('s'):
            save_path = f'results/squat_{squat_state}_{squat_count}rep.png'
            cv2.imwrite(save_path, frame)
            print(f'저장됨: {save_path}')

        # 단계별 자동 저장 (정면/측면 구분)
        if squat_state == "bottom":
            save_path = f'results/auto_bottom_{view}_{squat_count}rep.png'
            if not os.path.exists(save_path):
                cv2.imwrite(save_path, frame)
                print(f'최저점 자동 저장: {save_path}')

cap.release()
cv2.destroyAllWindows()