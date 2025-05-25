import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2 as cv
import numpy as np
import time

# STEP 1: Create the HandLandmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)

# MediaPipe connections (21 hand landmarks)
mp_hands = mp.solutions.hands
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS

# Function to draw landmarks and connections
def draw_landmarks_on_image(image, detection_result):
    for hand_landmarks in detection_result.hand_landmarks:
        landmark_points = []
        for landmark in hand_landmarks:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            landmark_points.append((x, y))
            cv.circle(image, (x, y), 5, (0, 255, 0), -1)

        # Draw connections
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                cv.line(image, landmark_points[start_idx], landmark_points[end_idx], (255, 0, 0), 2)

    return image

# STEP 2: Start webcam feed
capture = cv.VideoCapture(0)
capture.set(cv.CAP_PROP_FRAME_WIDTH, 1280)   # Set width
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720)   # Set height

prev_time = 0

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break

    # Resize to improve performance
    frame = cv.resize(frame, (640, 480))

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    # Print hand landmark coordinates
    if detection_result.hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            print(f"Hand {hand_idx + 1}:")
            for idx, landmark in enumerate(hand_landmarks):
                print(f"  Landmark {idx}: x={landmark.x:.3f}, y={landmark.y:.3f}, z={landmark.z:.3f}")

    annotated_frame = draw_landmarks_on_image(frame.copy(), detection_result)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv.putText(annotated_frame, f'FPS: {int(fps)}', (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display
    cv.imshow('Hand Landmarks', annotated_frame)

    if cv.waitKey(1) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
