import cv2
import mediapipe as mp
import numpy as np
import winsound  # For audio alert on Windows

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Constants
KNOWN_WIDTH = 8.0  # cm, average hand width (adjust based on your hand or target)
FOCAL_LENGTH = 500  # Placeholder; calibrate for your camera (see below)


def calibrate_focal_length(known_distance, known_width, measured_width):
    """Calibrate focal length using a known distance and measured pixel width."""
    return (measured_width * known_distance) / known_width


def calculate_distance(perceived_width):
    """Calculate distance from camera in cm."""
    if perceived_width <= 0:  # Avoid division by zero
        return float('inf')  # Return infinity for invalid width
    return (KNOWN_WIDTH * FOCAL_LENGTH) / perceived_width


# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Optional: Calibration step (uncomment to use)
# Hold your hand at a known distance (e.g., 30 cm) and measure width
# KNOWN_DISTANCE = 30.0  # cm
# print("Hold hand at 30 cm from camera for calibration. Press 'c' to calibrate.")
# while True:
#     ret, img = cap.read()
#     if not ret:
#         break
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = hands.process(img_rgb)
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             x_min, x_max = float('inf'), 0
#             for lm in hand_landmarks.landmark:
#                 x = int(lm.x * img.shape[1])
#                 x_min, x_max = min(x_min, x), max(x_max, x)
#             perceived_width = x_max - x_min
#             cv2.putText(img, "Hold hand at 30 cm, press 'c'", (50, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#             mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#     cv2.imshow('Calibration', img)
#     if cv2.waitKey(1) & 0xFF == ord('c'):
#         FOCAL_LENGTH = calibrate_focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, perceived_width)
#         print(f"Calibrated Focal Length: {FOCAL_LENGTH}")
#         break

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        print("Error: Failed to grab frame")
        break

    # Convert to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box coordinates
            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            # Calculate perceived width and distance
            perceived_width = x_max - x_min
            distance = calculate_distance(perceived_width)

            # Alert if distance is below 20 cm
            if distance <= 20 and distance > 0:
                winsound.Beep(1000, 300)  # 1000 Hz, 300 ms beep

            # Draw rectangle and distance text
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(img, f'Distance: {distance:.2f} cm', (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Draw hand landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Hand Distance Measurement', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()