import cv2 as cv
import numpy as np
import mediapipe as mp
import serial
import time

def calculate_angle(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    
    # Handle division by zero cases
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 90.0  # Return neutral position if vectors are zero
    
    cosine = np.dot(a, b) / (norm_a * norm_b)
    cosine = np.clip(cosine, -1.0, 1.0)  # Ensure valid range for arccos
    return np.degrees(np.arccos(cosine))

# Initialize serial connection
try:
    arduino = serial.Serial('COM14', 9600)
    time.sleep(2)
except serial.SerialException as e:
    print(f"Serial connection error: {e}")
    arduino = None

# MediaPipe setup
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

# Finger configuration
finger_indices = [(1, 2, 3), (5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19)]
finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

# Servo types: True for degree-based (D3, D4), False for state-based
servo_types = [False, True, True, False, False]  # Index 1 and 2 are degree-based

# Angle thresholds
min_angle = 30   # Fully extended
max_angle = 150  # Fully bent
extended_threshold = 80  # Below this is considered open
bent_threshold = 100    # Above this is considered closed

# Finger states - only open/closed for state-based servos
finger_states = ['open'] * 5
last_valid_angles = [90.0] * 5  # Start at neutral position
last_sent_angles = [90.0] * 5   # Track last sent angles for degree-based servos
last_sent_states = ['open'] * 5  # Track last sent states to Arduino to prevent duplicates

def map_angle_to_servo(angle, finger_index):
    """Map finger angle to servo angle based on servo type"""
    if servo_types[finger_index]:  # Degree-based servo (D3, D4)
        # Map finger angle (30-150) to servo angle (0-180)
        # Inverted mapping: smaller finger angle = larger servo angle
        servo_angle = int(np.interp(angle, [min_angle, max_angle], [180, 0]))
        return max(0, min(180, servo_angle))
    else:  # State-based servo - only open/closed
        if angle > bent_threshold:
            return 180  # Closed
        else:
            return 0   # Open (default for extended and in-between)

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
            # Draw hand landmarks
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
                    
                    # Validate angle
                    if np.isnan(angle):
                        angle = last_valid_angles[i]
                    else:
                        angle = np.clip(angle, min_angle, max_angle)
                        last_valid_angles[i] = angle

                    # Determine finger state - only open/closed for state-based servos
                    if servo_types[i]:  # Degree-based servo
                        # For degree-based servos, we don't need state tracking
                        new_state = 'degree_controlled'
                    else:  # State-based servo
                        if angle > bent_threshold:
                            new_state = 'closed'
                        else:
                            new_state = 'open'

                    # Get servo angle
                    servo_angle = map_angle_to_servo(angle, i)
                    
                    # Send command based on servo type
                    if servo_types[i]:  # Degree-based servo (D3, D4)
                        # Send if angle changed significantly (reduce noise)
                        angle_diff = abs(servo_angle - last_sent_angles[i])
                        if angle_diff > 3:  # Only send if change is > 3 degrees
                            if arduino is not None:
                                data = f"{i},angle,{servo_angle}\n"
                                try:
                                    arduino.write(data.encode())
                                    last_sent_angles[i] = servo_angle
                                    print(f"Sent: Finger {i} angle {servo_angle}°")
                                except serial.SerialException as e:
                                    print(f"Serial write error: {e}")
                    else:  # State-based servo
                        # Only send when state actually changes AND different from last sent state
                        if new_state != finger_states[i] and new_state != last_sent_states[i]:
                            finger_states[i] = new_state
                            last_sent_states[i] = new_state  # Record what we sent
                            
                            if arduino is not None:
                                data = f"{i},state,{servo_angle}\n"
                                try:
                                    arduino.write(data.encode())
                                    print(f"Sent: Finger {i} state {new_state} (servo_angle: {servo_angle})")
                                except serial.SerialException as e:
                                    print(f"Serial write error: {e}")
                        else:
                            # Update local state even if not sending to Arduino
                            finger_states[i] = new_state

                    # Visual feedback
                    cv.putText(frame, f"{finger_names[i]}: {int(angle)}°", 
                              (10, 30 + i * 30), cv.FONT_HERSHEY_SIMPLEX, 
                              0.7, (0, 255, 255), 2)
                    
                    # Show servo type and value
                    if servo_types[i]:
                        servo_info = f"Servo: {servo_angle}° (Last: {int(last_sent_angles[i])}°)"
                        state_color = (255, 0, 0)  # Blue for degree-based
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

# Cleanup
cap.release()
cv.destroyAllWindows()
if arduino is not None:
    arduino.close()