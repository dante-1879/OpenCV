import cv2 as cv
import numpy as np
import mediapipe as mp
from collections import deque, Counter

# Load gesture labels
with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_labels = [row.strip() for row in f.readlines()]

with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
    point_labels = [row.strip() for row in f.readlines()]

# Initialize MediaPipe hands and drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Point history buffer
point_history = deque(maxlen=16)
gesture_history = deque(maxlen=16)

# Load your custom gesture classifiers
from model import KeyPointClassifier, PointHistoryClassifier
keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()

# Start webcam
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.flip(frame, 1)
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture_text = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w = frame.shape[:2]
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

            # Draw hand landmarks and connections
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4),
            )

            # Preprocess for keypoint classification
            base_x, base_y = landmarks[0]
            relative_landmarks = [(x - base_x, y - base_y) for x, y in landmarks]
            flattened = np.array(relative_landmarks).flatten()
            norm = np.linalg.norm(flattened)
            normalized = flattened / norm if norm != 0 else flattened

            # Classify static hand sign
            gesture_id = keypoint_classifier(normalized.tolist())
            gesture_text = keypoint_labels[gesture_id]

            # Update point history if gesture is pointing (id == 2)
            if gesture_id == 2:
                point_history.append(landmarks[8])  # index fingertip
            else:
                point_history.append((0, 0))

            # Dynamic gesture classification
            if len(point_history) == 16:
                base_px, base_py = point_history[0]
                rel_history = [(x - base_px, y - base_py) for x, y in point_history]
                history_flat = np.array(rel_history).flatten() / np.array([w, h] * 16)
                dyn_id = point_history_classifier(history_flat.tolist())
                gesture_history.append(dyn_id)
                most_common_dyn = Counter(gesture_history).most_common(1)[0][0]
                gesture_text += f" + {point_labels[most_common_dyn]}"

    # Show gesture text
    if gesture_text:
        cv.putText(frame, gesture_text, (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv.imshow('Gesture', frame)
    if cv.waitKey(10) == 27:
        break

cap.release()
cv.destroyAllWindows()
