import os
import cv2
import pickle
import threading
from queue import Queue
from feat import Detector
import time

def process_snapshot(temp_image_path, frame, face_tracker, detector, model, result_queue):
    """
    Processes a snapshot in a separate thread to prevent blocking the webcam feed.
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


def open_webcam(result_queue, temp_image_path, face_tracker, detector, model, interval=15, stop_event=None):
    """
    Opens the webcam and captures frames every specified interval in a background thread.
    """
    # Open the laptop webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Webcam is running in the background. Press Ctrl+C to stop.")

    last_capture_time = 0  # To track the interval
    while not stop_event.is_set():  # Check the stop event for termination
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If the frame was not captured correctly, break the loop
        if not ret:
            print("Failed to grab frame.")
            break

        # Display the live feed
        cv2.imshow("Live Feed", frame)

        # Capture and process image every interval seconds
        current_time = time.time()
        if current_time - last_capture_time >= interval:
            # Process snapshot in a separate thread
            threading.Thread(
                target=process_snapshot,
                args=(temp_image_path, frame.copy(), face_tracker, detector, model, result_queue)
            ).start()
            last_capture_time = current_time

        # Close the live feed window if 'ESC' is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release the capture and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Initialize resources
    result_queue = Queue()
    temp_image_path = "temp_face.jpg"
    face_tracker = cv2.CascadeClassifier("frontal_face_features.xml")
    detector = Detector()

    # Load the model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Stop event to gracefully terminate the webcam thread
    stop_event = threading.Event()

    # Run open_webcam in a background thread
    webcam_thread = threading.Thread(
        target=open_webcam,
        args=(result_queue, temp_image_path, face_tracker, detector, model, 15, stop_event)
    )
    webcam_thread.start()

    try:
        while True:
            # Retrieve results from the queue in the main thread
            if not result_queue.empty():
                result = result_queue.get()
                if result is not None:
                    print(f"Detected Emotion: {result['emotion']}, Confidence: {result['confidence']:.2f}")
                else:
                    print("No result received.")
            time.sleep(1)  # Simulate other main thread tasks
    except KeyboardInterrupt:
        print("Stopping the webcam thread...")
        stop_event.set()  # Signal the webcam thread to stop
        webcam_thread.join()  # Wait for the thread to finish
        print("Webcam thread stopped.")
