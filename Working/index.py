import cv2 as cv
import numpy as np
import mediapipe as mp
import serial
import time

def calculate_angle(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 90.0
    
    cosine = np.dot(a, b) / (norm_a * norm_b)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))

try:
    arduino = serial.Serial('COM14', 9600)
    time.sleep(2)
except serial.SerialException as e:
    print(f"Serial connection error: {e}")
    arduino = None

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error opening video capture")
    exit()

finger_indices = [(1, 2, 3), (5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19)]
finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
servo_types = [False, True, True, False, False]

min_angle = 30
max_angle = 150
extended_threshold = 80
bent_threshold = 100

finger_states = ['open'] * 5
last_valid_angles = [90.0] * 5
last_sent_angles = [90.0] * 5
last_sent_states = ['open'] * 5

def map_angle_to_servo(angle, finger_index):
    if servo_types[finger_index]:
        servo_angle = int(np.interp(angle, [min_angle, max_angle], [180, 0]))
        return max(0, min(180, servo_angle))
    else:
        if finger_index == 3 or finger_index == 4:  # Ring and pinky - flipped logic
            return 0 if angle > bent_threshold else 180
        else:
            return 180 if angle > bent_threshold else 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    
    frame = cv.flip(frame, 1)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            h, w = frame.shape[:2]
            lm = [(int(landmark.x * w), int(landmark.y * h)) 
                 for landmark in hand_landmarks.landmark]

            for i, (a, b, c) in enumerate(finger_indices):
                try:
                    angle = calculate_angle(lm[a], lm[b], lm[c])
                    
                    if np.isnan(angle):
                        angle = last_valid_angles[i]
                    else:
                        angle = np.clip(angle, min_angle, max_angle)
                        last_valid_angles[i] = angle

                    if servo_types[i]:
                        new_state = 'degree_controlled'
                    else:
                        if i == 3 or i == 4:  # Flipped logic for ring and pinky
                            new_state = 'closed' if angle <= bent_threshold else 'open'
                        else:
                            new_state = 'closed' if angle > bent_threshold else 'open'

                    servo_angle = map_angle_to_servo(angle, i)
                    
                    if servo_types[i]:
                        angle_diff = abs(servo_angle - last_sent_angles[i])
                        if angle_diff > 3:
                            if arduino is not None:
                                data = f"{i},angle,{servo_angle}\n"
                                try:
                                    arduino.write(data.encode())
                                    last_sent_angles[i] = servo_angle
                                    print(f"Sent: Finger {i} angle {servo_angle}°")
                                except serial.SerialException as e:
                                    print(f"Serial write error: {e}")
                    else:
                        if new_state != finger_states[i] and new_state != last_sent_states[i]:
                            finger_states[i] = new_state
                            last_sent_states[i] = new_state
                            
                            if arduino is not None:
                                data = f"{i},state,{servo_angle}\n"
                                try:
                                    arduino.write(data.encode())
                                    print(f"Sent: Finger {i} state {new_state} (servo_angle: {servo_angle})")
                                except serial.SerialException as e:
                                    print(f"Serial write error: {e}")
                        else:
                            finger_states[i] = new_state

                    cv.putText(frame, f"{finger_names[i]}: {int(angle)}°", 
                              (10, 30 + i * 30), cv.FONT_HERSHEY_SIMPLEX, 
                              0.7, (0, 255, 255), 2)
                    
                    if servo_types[i]:
                        servo_info = f"Servo: {servo_angle}° (Last: {int(last_sent_angles[i])}°)"
                        state_color = (255, 0, 0)
                    else:
                        servo_info = f"State: {new_state} (Sent: {last_sent_states[i]})"
                        state_color = (0, 255, 0) if new_state == 'closed' else (0, 0, 255)
                    
                    cv.putText(frame, servo_info, 
                              (250, 30 + i * 30), cv.FONT_HERSHEY_SIMPLEX, 
                              0.7, state_color, 2)

                except IndexError:
                    print(f"Landmark index error for finger {i}")
                    continue

    cv.imshow("Finger Control", frame)
    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
if arduino is not None:
    arduino.close()