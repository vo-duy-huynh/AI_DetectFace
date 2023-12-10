import cv2
import tensorflow as tf
import numpy as np
import os

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained face recognition model
face_recognition_model = tf.keras.models.load_model('modelface.h5')

# Load and preprocess the image
filename = r'D:\NCKH\AI\Data\Faces\testing\huynhpro\huynhpro.jpg'
image = cv2.imread(filename)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform face detection
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# lấy ra danh sách folder trong thư mục training
list_folder = os.listdir('Data/Faces/training')

# Set a confidence threshold (adjust as needed)
confidence_threshold = 0.7

# Process each detected face
for (x, y, w, h) in faces:
    # Extract the face region
    face_roi = image[y:y+h, x:x+w]

    # Resize the face image to the required input size of the model
    face_roi = cv2.resize(face_roi, (100, 100))

    # Normalize pixel values to be between 0 and 1
    face_roi = face_roi / 255.0

    # Reshape the face image to match the input shape of the model
    face_roi = np.reshape(face_roi, (1, 100, 100, 3))

    # Perform face recognition using the loaded model
    predictions = face_recognition_model.predict(face_roi)

    # Get the predicted label and confidence
    predicted_label = np.argmax(predictions)
    confidence = predictions[0][predicted_label]
    print(confidence)
    # Check if the confidence is above the threshold
    if confidence > confidence_threshold:
        label = list_folder[predicted_label]
    else:
        label = "unknown"

    # Draw a rectangle around the detected face and display the predicted label
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 0, 0), 2)

# Display the result
cv2.imshow('Face Recognition Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
