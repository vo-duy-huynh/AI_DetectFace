import cv2
import tensorflow as tf
import numpy as np
import os
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# đọc video từ file
video_capture = cv2.VideoCapture(1)
list_folder = os.listdir('Data/Faces/training')
face_recognition_model = tf.keras.models.load_model('modelface.h5')
while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    confidence_threshold = 0.7
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        expected_width, expected_height = 100, 100
        face_roi = cv2.resize(face_roi, (expected_width, expected_height))
        face_roi = face_roi.astype("float") / 255.0
        face_roi = np.reshape(
            face_roi, (1, expected_width, expected_height, 3))
        predictions = face_recognition_model.predict(face_roi)
        predicted_label = np.argmax(predictions)
        confidence = predictions[0][predicted_label]
        if confidence > confidence_threshold:
            label = list_folder[predicted_label]
        else:
            label = "unknown"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
    cv2.imshow('Face Recognition Result', frame)
    if cv2.waitKey(1) == 27:
        break
video_capture.release()
cv2.destroyAllWindows()
