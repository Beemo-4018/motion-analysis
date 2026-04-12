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
    x_diff = abs(landmarks[11].x - landmarks[12].x)
    return "front" if x_diff > 0.08 else "side"

# 상태 변수
squat_count     = 0
prev_knee_angle = None
prev_state      = None
DELTA_THRESHOLD = 1.5

def classify_state(knee_angle, delta):
    if knee_angle > 160:
        return "standing"
    elif knee_angle < 110:
        return "bottom"
    elif delta < -DELTA_THRESHOLD:
        return "descending"
    elif delta > DELTA_THRESHOLD:
        return "ascending"
    else:
        return "descending" if knee_angle > 135 else "ascending"

def analyze_side(landmarks, frame):
    global squat_count, prev_knee_angle, prev_state

    hip      = [landmarks[24].x, landmarks[24].y]
    knee     = [landmarks[26].x, landmarks[26].y]
    ankle    = [landmarks[28].x, landmarks[28].y]
    shoulder = [landmarks[12].x, landmarks[12].y]

    knee_angle  = calculate_angle(hip, knee, ankle)
    trunk_angle = calculate_angle(shoulder, hip, knee)

    delta = knee_angle - prev_knee_angle if prev_knee_angle is not None else 0.0
    prev_knee_angle = knee_angle

    state = classify_state(knee_angle, delta)

    # 카운팅
    if state == "standing" and prev_state == "ascending":
        squat_count += 1
    prev_state = state

    state_color = {
        "standing":   (0, 255, 0),
        "descending": (0, 255, 255),
        "bottom":     (0, 165, 255),
        "ascending":  (255, 165, 0)
    }
    color = state_color[state]

    cv2.putText(frame, 'Mode: side',
                (50, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f'Knee: {knee_angle}',
                (50, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f'Delta: {round(delta, 1)}',
                (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f'State: {state}',
                (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f'Count: {squat_count}',
                (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, f'Trunk: {trunk_angle}',
                (50, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if trunk_angle < 50:
        cv2.putText(frame, 'WARNING: Trunk Forward',
                    (50, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return frame

def analyze_front(landmarks, frame):
    global squat_count, prev_knee_angle, prev_state

    l_hip   = [landmarks[23].x, landmarks[23].y]
    l_knee  = [landmarks[25].x, landmarks[25].y]
    l_ankle = [landmarks[27].x, landmarks[27].y]
    r_hip   = [landmarks[24].x, landmarks[24].y]
    r_knee  = [landmarks[26].x, landmarks[26].y]
    r_ankle = [landmarks[28].x, landmarks[28].y]

    l_angle   = calculate_angle(l_hip, l_knee, l_ankle)
    r_angle   = calculate_angle(r_hip, r_knee, r_ankle)
    avg_angle = (l_angle + r_angle) / 2
    diff      = abs(l_angle - r_angle)

    delta = avg_angle - prev_knee_angle if prev_knee_angle is not None else 0.0
    prev_knee_angle = avg_angle

    state = classify_state(avg_angle, delta)

    # 카운팅
    if state == "standing" and prev_state == "ascending":
        squat_count += 1
    prev_state = state

    state_color = {
        "standing":   (0, 255, 0),
        "descending": (0, 255, 255),
        "bottom":     (0, 165, 255),
        "ascending":  (255, 165, 0)
    }
    color = state_color[state]

    cv2.putText(frame, 'Mode: front',
                (50, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f'L Knee: {l_angle}',
                (50, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f'R Knee: {r_angle}',
                (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f'Delta: {round(delta, 1)}',
                (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f'State: {state}',
                (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f'Count: {squat_count}',
                (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    if diff > 15:
        cv2.putText(frame, f'WARNING: Asymmetry {round(diff, 1)}',
                    (50, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

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

        cv2.imshow('Squat Analyzer v2', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_path = f'results/squat_v2_{int(cv2.getTickCount())}.png'
            cv2.imwrite(save_path, frame)
            print(f'저장됨: {save_path}')

cap.release()
cv2.destroyAllWindows()