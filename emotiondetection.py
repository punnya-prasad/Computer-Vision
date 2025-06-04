import cv2
from deepface import DeepFace

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, img = cap.read()
    if not ret:
        print("Error: Failed to grab frame")
        break

    try:
        # Analyze emotion only
        results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        print("Full Results:", results)
        emotion = results[0]['dominant_emotion']
        print("Detected Emotion:", emotion)
    except Exception as e:
        print("Error in DeepFace analysis:", e)
        emotion = "Unknown"

    cv2.putText(img, f'Emotion: {emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Emotion Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()