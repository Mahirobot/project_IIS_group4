import os
import cv2
import pickle
import threading
from queue import Queue
from feat import Detector

def capture_and_process_image(temp_image_path, frame, face_tracker, detector, model, result_queue):
    """
    Captures an image from the frame, saves it temporarily, and processes it using the detector.
    """
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_tracker.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        print("No face detected.")
        result_queue.put(None)  # Indicate no result
        return

    # Process the first detected face
    x, y, w, h = faces[0]
    face_roi = frame[y:y + h, x:x + w]

    # Save the face ROI as a temporary image
    cv2.imwrite(temp_image_path, face_roi)

    try:
        # Pass the image path to the detector
        result = detector.detect_image(temp_image_path)
        face_features = result.aus.values.flatten()

        # Predict the emotion using the preloaded model
        emotion = model.predict([face_features])[0]
        confidence = model.predict_proba([face_features]).max()

        print(f"Emotion: {emotion}, Confidence: {confidence:.2f}")
        # Return the result via the queue
        result_queue.put({'emotion': emotion, 'confidence': confidence})
    except Exception as e:
        print(f"Error during detection: {e}")
        result_queue.put(None)  # Indicate no result
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


def open_webcam():
    face_tracker = cv2.CascadeClassifier("frontal_face_features.xml")
    detector = Detector()

    # Preload the Random Forest model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Open the laptop webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        exit()

    print("Press 'Enter' to capture and process an image.")
    print("Press 'ESC' to exit.")

    temp_image_path = "temp_face.jpg"  # Path for saving the snapshot

    # Queue to store results from the thread
    result_queue = Queue()

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
            # Use threading to avoid blocking the video feed
            threading.Thread(
                target=capture_and_process_image,
                args=(temp_image_path, frame.copy(), face_tracker, detector, model, result_queue)
            ).start()

        # Check if there is a result in the queue
        if not result_queue.empty():
            result = result_queue.get()  # Get the result from the queue
            if result is not None:
                print(f"Detected Emotion: {result['emotion']}, Confidence: {result['confidence']:.2f}")
            else:
                print("No result received.")

    # Release the capture and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    open_webcam()
