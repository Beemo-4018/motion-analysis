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

def detect_view(landmarks):
    left_shoulder  = landmarks[11]
    right_shoulder = landmarks[12]
    x_diff = abs(left_shoulder.x - right_shoulder.x)
    if x_diff > 0.08:
        return "front"
    else:
        return "side"

# 상태 변수
shoulder_max_angle = 0
hold_start_time    = None
hold_duration      = 0.0
HOLD_THRESHOLD     = 140
HOLD_TARGET        = 15.0  # 목표 유지시간 (초)

# 정면용 상태 변수
front_hold_start_time = None
front_hold_duration   = 0.0
front_sound_played    = False

def analyze_side(landmarks, frame):
    global shoulder_max_angle, hold_start_time, hold_duration

    # 어느 쪽 팔이 더 올라가 있는지 감지
    l_wrist_y = landmarks[15].y
    r_wrist_y = landmarks[16].y

    # y좌표가 작을수록 위에 있음
    if l_wrist_y < r_wrist_y:
        # 왼쪽 팔이 올라가 있음
        wrist    = [landmarks[15].x, landmarks[15].y]
        shoulder = [landmarks[11].x, landmarks[11].y]
        hip      = [landmarks[23].x, landmarks[23].y]
        ear      = [landmarks[7].x,  landmarks[7].y]
        side_label = 'L'
    else:
        # 오른쪽 팔이 올라가 있음
        wrist    = [landmarks[16].x, landmarks[16].y]
        shoulder = [landmarks[12].x, landmarks[12].y]
        hip      = [landmarks[24].x, landmarks[24].y]
        ear      = [landmarks[8].x,  landmarks[8].y]
        side_label = 'R'

    shoulder_angle = calculate_angle(wrist, shoulder, hip)
    shoulder_angle = min(round(shoulder_angle * 1.04, 1), 180.0)
    trunk_angle    = calculate_angle(ear, shoulder, hip)

    if shoulder_angle > shoulder_max_angle:
        shoulder_max_angle = shoulder_angle

    if shoulder_angle >= HOLD_THRESHOLD:
        if hold_start_time is None:
            hold_start_time = time.time()
            # 소리 재생 플래그 초기화
        hold_duration = round(time.time() - hold_start_time, 1)

        # 15초 달성 시 소리
        if hold_duration >= HOLD_TARGET and hold_duration - 0.1 < HOLD_TARGET:
            os.system('afplay /System/Library/Sounds/Glass.aiff &')
    else:
        hold_start_time = None

    trunk_warning = trunk_angle < 150

    if shoulder_angle < 60:
        color = (0, 255, 0)
    elif shoulder_angle < 120:
        color = (0, 255, 255)
    else:
        color = (0, 165, 255)

    cv2.putText(frame, f'Mode: side ({side_label})',
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 2)
    cv2.putText(frame, f'Shoulder: {shoulder_angle}',
                (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2)
    cv2.putText(frame, f'Trunk: {trunk_angle}',
                (50, 110), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)
    cv2.putText(frame, f'Max: {shoulder_max_angle}',
                (50, 140), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    hold_color = (0, 255, 0) if hold_duration >= HOLD_TARGET else (255, 200, 0)

    cv2.putText(frame, f'Hold: {hold_duration}s / {HOLD_TARGET}s',
                (50, 170), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, hold_color, 2)

    if hold_duration >= HOLD_TARGET:
        cv2.putText(frame, 'GOAL REACHED!',
                    (50, 210), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 3)

    if trunk_warning:
        cv2.putText(frame, 'WARNING: Trunk Compensation',
                    (50, 240), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

    return frame

def analyze_front(landmarks, frame):
    global front_hold_start_time, front_hold_duration, front_sound_played

    l_wrist    = [landmarks[15].x, landmarks[15].y]
    l_shoulder = [landmarks[11].x, landmarks[11].y]
    l_hip      = [landmarks[23].x, landmarks[23].y]

    r_wrist    = [landmarks[16].x, landmarks[16].y]
    r_shoulder = [landmarks[12].x, landmarks[12].y]
    r_hip      = [landmarks[24].x, landmarks[24].y]

    l_angle = calculate_angle(l_wrist, l_shoulder, l_hip)
    r_angle = calculate_angle(r_wrist, r_shoulder, r_hip)

    # 더 올라간 팔 기준
    l_wrist_y = landmarks[15].y
    r_wrist_y = landmarks[16].y

    if l_wrist_y < r_wrist_y:
        active_label = 'L'
        active_angle = l_angle
    else:
        active_label = 'R'
        active_angle = r_angle

    active_color = (0, 165, 255) if active_angle >= 120 else (0, 255, 255) if active_angle >= 60 else (0, 255, 0)
    diff = abs(l_angle - r_angle)

    # 유지시간 측정 (90도 이상)
    if active_angle >= 90:
        if front_hold_start_time is None:
            front_hold_start_time = time.time()
            front_sound_played = False
        front_hold_duration = round(time.time() - front_hold_start_time, 1)

        # 15초 달성 시 소리
        if front_hold_duration >= HOLD_TARGET and not front_sound_played:
            os.system('afplay /System/Library/Sounds/Glass.aiff &')
            front_sound_played = True
    else:
        front_hold_start_time = None
        front_hold_duration   = 0.0
        front_sound_played    = False

    # 목표 달성 여부
    hold_color = (0, 255, 0) if front_hold_duration >= HOLD_TARGET else (255, 200, 0)

    # Sidebending 감지
    shoulder_y_diff  = abs(landmarks[11].y - landmarks[12].y)
    sidebend_warning = shoulder_y_diff > 0.08

    # 화면 표시
    cv2.putText(frame, 'Mode: front',
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 2)
    cv2.putText(frame, f'{active_label} Shoulder: {active_angle}',
                (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, active_color, 2)
    cv2.putText(frame, f'L: {l_angle}  R: {r_angle}',
                (50, 110), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Diff: {round(diff, 1)}',
                (50, 140), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    cv2.putText(frame, f'Hold: {front_hold_duration}s / {HOLD_TARGET}s',
                (50, 170), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, hold_color, 2)

    if front_hold_duration >= HOLD_TARGET:
        cv2.putText(frame, 'GOAL REACHED!',
                    (50, 210), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 3)

    if diff > 15:
        cv2.putText(frame, f'WARNING: Asymmetry {round(diff, 1)}',
                    (50, 250), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

    if sidebend_warning:
        cv2.putText(frame, 'WARNING: Trunk Sidebending',
                    (50, 280), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

    return frame

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
            view = detect_view(landmarks)

            if view == "side":
                frame = analyze_side(landmarks, frame)
            else:
                frame = analyze_front(landmarks, frame)

            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # 최대각도 자동 저장 (측면)
            if view == "side":
                # 최대각도 갱신 시 저장
                save_path = f'results/shoulder_side_{shoulder_max_angle}deg.png'
                if not os.path.exists(save_path):
                    cv2.imwrite(save_path, frame)
                    print(f'저장됨: {save_path}')

                # GOAL REACHED 시 저장
                if hold_duration >= HOLD_TARGET:
                    goal_path = f'results/shoulder_side_goal_{shoulder_max_angle}deg.png'
                    if not os.path.exists(goal_path):
                        cv2.imwrite(goal_path, frame)
                        print(f'목표 달성 저장됨: {goal_path}')

            # 정면 자동 저장 (30프레임마다)
            elif view == "front":
                if int(cv2.getTickCount()) % 30 == 0:
                    save_path = f'results/shoulder_front_{int(cv2.getTickCount())}.png'
                    cv2.imwrite(save_path, frame)
                    print(f'정면 저장됨: {save_path}')

        cv2.imshow('Shoulder Ladder', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f'최대 굴곡각도: {shoulder_max_angle}')
            break
        elif key == ord('s'):
            save_path = f'results/shoulder_{view}_{shoulder_max_angle}deg.png'
            cv2.imwrite(save_path, frame)
            print(f'수동 저장됨: {save_path}')

cap.release()
cv2.destroyAllWindows()