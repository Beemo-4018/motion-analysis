import cv2
import mediapipe as mp
import numpy as np
import os
import time

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

# 상태 변수
hold_start_time  = None
hold_duration    = 0.0
max_angle        = 0.0
sound_played     = False
HOLD_TARGET      = 15.0
ANGLE_THRESHOLD  = 30  # 30도 이상 들어올리면 측정 시작

def analyze_slr(landmarks, frame):
    global hold_start_time, hold_duration, max_angle, sound_played

    # 들어올린 다리 자동 감지
    # 누운 자세에서 y좌표가 작을수록 위로 올라간 것
    l_ankle_y = landmarks[27].y
    r_ankle_y = landmarks[28].y
    l_hip_y   = landmarks[23].y
    r_hip_y   = landmarks[24].y

    # 엉덩이 대비 발목이 얼마나 올라갔는지로 판단
    l_lift = l_hip_y - l_ankle_y
    r_lift = r_hip_y - r_ankle_y

    if l_lift > r_lift:
        hip      = [landmarks[23].x, landmarks[23].y]
        knee     = [landmarks[25].x, landmarks[25].y]
        ankle    = [landmarks[27].x, landmarks[27].y]
        shoulder = [landmarks[11].x, landmarks[11].y]
        leg_label = 'L'
    else:
        hip      = [landmarks[24].x, landmarks[24].y]
        knee     = [landmarks[26].x, landmarks[26].y]
        ankle    = [landmarks[28].x, landmarks[28].y]
        shoulder = [landmarks[12].x, landmarks[12].y]
        leg_label = 'R'

    # 다리 각도 (엉덩이-무릎-발목 기준)
    leg_angle = calculate_angle(hip, knee, ankle)
    # 수정 - 누운 자세에서 실제 들어올린 각도
    # 엉덩이-발목 선이 바닥과 이루는 각도로 계산
    if leg_label == 'L':
        shoulder = [landmarks[11].x, landmarks[11].y]
        hip      = [landmarks[23].x, landmarks[23].y]
        ankle    = [landmarks[27].x, landmarks[27].y]
    else:
        shoulder = [landmarks[12].x, landmarks[12].y]
        hip      = [landmarks[24].x, landmarks[24].y]
        ankle    = [landmarks[28].x, landmarks[28].y]

    lift_angle = calculate_angle(shoulder, hip, ankle)
    lift_angle = round(180.0 - lift_angle, 1)
    
    # 최대각도 갱신
    if lift_angle > max_angle:
        max_angle = lift_angle

    # 골반 안정성 (양쪽 엉덩이 y좌표 차이)
    hip_y_diff = abs(landmarks[23].y - landmarks[24].y)
    pelvis_warning = hip_y_diff > 0.05

    # 유지시간 측정
    if lift_angle >= ANGLE_THRESHOLD:
        if hold_start_time is None:
            hold_start_time = time.time()
            sound_played = False  # 다리 바뀔 때마다 초기화
        hold_duration = round(time.time() - hold_start_time, 1)

        if hold_duration >= HOLD_TARGET and not sound_played:
            os.system('afplay /System/Library/Sounds/Glass.aiff &')
            sound_played = True
    else:
        hold_start_time = None
        hold_duration   = 0.0
        sound_played    = False  # ← 이게 핵심, 다리 내리면 무조건 초기화

    # 색상
    if lift_angle < 30:
        color = (0, 255, 0)
    elif lift_angle < 60:
        color = (0, 255, 255)
    else:
        color = (0, 165, 255)

    hold_color = (0, 255, 0) if hold_duration >= HOLD_TARGET else (255, 200, 0)

    # 화면 표시
    cv2.putText(frame, f'SLR Mode ({leg_label})',
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 2)
    cv2.putText(frame, f'Lift Angle: {lift_angle}',
                (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2)
    cv2.putText(frame, f'Max: {max_angle}',
                (50, 110), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    cv2.putText(frame, f'Hold: {hold_duration}s / {HOLD_TARGET}s',
                (50, 140), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, hold_color, 2)

    if hold_duration >= HOLD_TARGET:
        cv2.putText(frame, 'GOAL REACHED!',
                    (50, 175), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 3)

    if pelvis_warning:
        cv2.putText(frame, 'WARNING: Pelvis Tilt',
                    (50, 215), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

    return frame, leg_label

os.makedirs('results', exist_ok=True)
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

            frame, leg_label = analyze_slr(landmarks, frame)

            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # GOAL REACHED 시 자동 저장
            if hold_duration >= HOLD_TARGET:
                goal_path = f'results/slr_goal_{leg_label}_{max_angle}deg.png'
                if not os.path.exists(goal_path):
                    cv2.imwrite(goal_path, frame)
                    print(f'목표 달성 저장됨: {goal_path}')

            # 최대각도 갱신 시 저장
            save_path = f'results/slr_max_{max_angle}deg.png'
            if not os.path.exists(save_path):
                cv2.imwrite(save_path, frame)
                print(f'최대각도 저장됨: {save_path}')

        cv2.imshow('SLR Analyzer', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f'최대 각도: {max_angle}')
            break
        elif key == ord('s'):
            save_path = f'results/slr_manual_{max_angle}deg.png'
            cv2.imwrite(save_path, frame)
            print(f'수동 저장됨: {save_path}')

cap.release()
cv2.destroyAllWindows()