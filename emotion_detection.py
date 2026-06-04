import cv2
from deepface import DeepFace

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

print("Starting Emotion Detection...")
print("Press 'Q' to quit.")

frame_count = 0

# Store latest emotion results
emotion = "Detecting..."
confidence = 0.0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    frame_count += 1

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    # Run DeepFace every 5 frames
    if frame_count % 5 == 0 and len(faces) > 0:

        # Use the largest detected face
        largest_face = max(faces, key=lambda f: f[2] * f[3])

        x, y, w, h = largest_face

        face_roi = frame[y:y+h, x:x+w]

        try:
            result = DeepFace.analyze(
                face_roi,
                actions=["emotion"],
                enforce_detection=False,
                silent=True
            )

            if isinstance(result, list):
                result = result[0]

            emotion = result["dominant_emotion"]
            confidence = result["emotion"][emotion]

        except Exception as e:
            print("DeepFace Error:", e)

    # Draw rectangles and labels
    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        label = f"{emotion} ({confidence:.1f}%)"

        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Show instructions
    cv2.putText(
        frame,
        "Press Q to Quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    cv2.imshow("Real-Time Facial Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("Emotion Detection Stopped.")