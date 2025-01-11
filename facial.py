# This is a sample Python script.

# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import pandas as pd
import os
import cv2
import pickle
from feat import Detector
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from PIL import Image

def capture_and_process_image(cap, face_tracker, detector):
    """
    Captures an image from the webcam, saves it temporarily, and processes it using the detector.
    """

    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        return None

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_tracker.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        print("No face detected.")
        return None

    # Process the first detected face (modify if you want to process multiple faces)
    x, y, w, h = faces[0]
    face_roi = frame[y:y + h, x:x + w]

    # Save the face ROI as a temporary image
    temp_image_path = "temp_face.jpg"
    cv2.imwrite(temp_image_path, face_roi)

    try:
        # Pass the image path to the detector
        result = detector.detect_image(temp_image_path)
        # Predict the emotion using the Random Forest model
        face_features = result.aus.values.flatten()
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        emotion = model.predict([face_features])[0]  # Model expects 2D array
        print(emotion)
            # Display the predicted emotion
        cv2.putText(frame, f'Emotion: {emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Delete the temporary image
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        return result
    except Exception as e:
        print(f"Error during detection: {e}")
        # Clean up the temporary file in case of error
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        return None


def open_webcam():
    face_tracker = cv2.CascadeClassifier("frontal_face_features.xml")
    detector = Detector()
    # Open the laptop webcam (default device is 0)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        exit()

    print("Press 'Enter' to capture and process an image.")
    print("Press 'ESC' to exit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If the frame was not captured correctly, break the loop
        if not ret:
            print("Failed to grab frame.")
            break

        # Display the live feed
        cv2.imshow("Live Feed", frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Press 'ESC' to exit the program
            break
        elif key == 13:  # Press 'Enter' to capture and process an image
            detection_result = capture_and_process_image(cap, face_tracker, detector)

    # Release the capture and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    open_webcam()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
