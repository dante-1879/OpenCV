import cv2 as cv
import numpy as np
import mediapipe as mp
import serial
import time

def angle_between(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def map_angle(angle, min_angle=30, max_angle=160):
    return int(np.interp(angle, [min_angle, max_angle], [0, 180]))

# Initialize serial connection to Arduino
arduino = serial.Serial('COM13', 9600)  # Change port as needed
time.sleep(2)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.flip(frame, 1)
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            h, w = frame.shape[:2]
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
            raw_landmarks = hand_landmarks.landmark  # For accessing .x directly

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Determine handedness (Left/Right)
            label = handedness.classification[0].label  # 'Left' or 'Right'

            # ---- Detect if thumb is open ----
            # Thumb is open if tip is to the side of joint depending on hand
            if label == 'Right':
                thumb_state = 0 if raw_landmarks[4].x < raw_landmarks[3].x else 1
            else:  # Left
                thumb_state = 0 if raw_landmarks[4].x > raw_landmarks[3].x else 1

            # Calculate angles for fingers
            index = angle_between(landmarks[5], landmarks[6], landmarks[7])
            middle = angle_between(landmarks[9], landmarks[10], landmarks[11])
            ring = angle_between(landmarks[13], landmarks[14], landmarks[15])
            pinky = angle_between(landmarks[17], landmarks[18], landmarks[19])
            fingers = [index, middle, ring, pinky]
            mapped_angles = [map_angle(f) for f in fingers]

            # Send data to Arduino
            data_str = f"{thumb_state}," + ",".join(map(str, mapped_angles)) + "\n"
            arduino.write(data_str.encode())

            # Show data
            cv.putText(frame, data_str.strip(), (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow('Hand to Arduino', frame)
    if cv.waitKey(10) == 27:  # ESC to exit
        break

cap.release()
cv.destroyAllWindows()
arduino.close()
