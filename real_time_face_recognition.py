import cv2
import numpy as np
import tensorflow as tf
import platform
import threading  # For non-blocking sound
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("facial_emotion_model.keras")

# Load OpenCV Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Function to preprocess face images
def preprocess_image(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (48, 48))  # Resize to match model input
    image = image.reshape(1, 48, 48, 1) / 255.0  # Normalize and reshape
    return image

# Function to play beep sound asynchronously
def beep_sound():
    system_os = platform.system()
    
    def play():
        if system_os == "Windows":
            import winsound
            winsound.Beep(1000, 500)  # Frequency 1000 Hz, Duration 500ms
        elif system_os == "Darwin":  # macOS
            import os
            os.system("afplay /System/Library/Sounds/Ping.aiff")  # Default macOS sound
        else:  # Linux
            import os
            os.system("aplay /usr/share/sounds/alsa/Front_Center.wav")  # Default Linux sound

    sound_thread = threading.Thread(target=play, daemon=True)  # Run in background
    sound_thread.start()

# Open webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set up full-screen window
cv2.namedWindow("Facial Emotion Recognition", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Facial Emotion Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

try:
    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]  # Extract face
            img = preprocess_image(face_img)  # Preprocess for CNN model

            # Predict emotion
            prediction = model.predict(img)
            emotion_label = labels[np.argmax(prediction)]

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Play beep sound asynchronously when "happy" is detected
            if emotion_label == "happy":
                beep_sound()

        # Display output in full-screen
        cv2.imshow("Facial Emotion Recognition", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Cleanup: Release webcam and close windows
    webcam.release()
    cv2.destroyAllWindows()
    print("Webcam released, windows closed.")
