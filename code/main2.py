import cv2
import tensorflow as tf
import numpy as np
import os

# Load the pre-trained face recognition model
face_recognition_model = tf.keras.models.load_model('modelface.h5')

# Open the camera
video_capture = cv2.VideoCapture(0)
list_folder = os.listdir('Data/Faces/training')

# Set the desired display size
display_width, display_height = 720, 720

while True:
    # Capture each frame from the camera
    ret, frame = video_capture.read()

    # Resize the frame to the required input size of the model
    expected_width, expected_height = 100, 100
    frame_resized = cv2.resize(frame, (expected_width, expected_height))

    # Normalize pixel values to be between 0 and 1
    frame_resized = frame_resized.astype("float") / 255.0

    # Reshape the frame to match the input shape of the model
    frame_input = np.reshape(
        frame_resized, (1, expected_width, expected_height, 3))

    # Perform face recognition using the loaded model
    predictions = face_recognition_model.predict(frame_input)

    # Get the predicted label and confidence
    predicted_label = np.argmax(predictions)
    confidence = predictions[0][predicted_label]

    # Check if the confidence is above the threshold
    confidence_threshold = 0.7
    if confidence > confidence_threshold:
        label = list_folder[predicted_label]
    else:
        label = "unknown"

    # Draw a square around the detected face
    faces = face_cascade.detectMultiScale(
        frame, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        # Ensure the square stays within the bounds of the frame
        x, y, w, h = max(0, x), max(0, y), min(
            w, display_width - x), min(h, display_height - y)

        # Draw the square
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the result
    frame_output = cv2.resize(frame, (display_width, display_height))
    # OpenCV uses BGR, convert to RGB for display
    frame_output = cv2.cvtColor(frame_output, cv2.COLOR_BGR2RGB)
    cv2.putText(frame_output, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 0, 0), 2)

    cv2.imshow('Face Recognition Result', frame_output)

    # Check for the 'Esc' key to exit the loop
    if cv2.waitKey(1) == 27:
        break

# Release the VideoCapture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
