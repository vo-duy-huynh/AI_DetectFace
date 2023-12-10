import cv2
import os
import time

save_folder = input("Nhập tên cần tạo ảnh: ")
os.makedirs('Data/Faces/training/' + save_folder, exist_ok=True)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 720)
face_count = 0
max_faces = 200
while cap.isOpened() and face_count < max_faces:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_roi = frame[y:y+h, x:x+w]
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_filename = os.path.join(
            'Data/Faces/training', save_folder, f'face_{face_count}.jpg')
        cv2.imwrite(face_filename, face_roi)
        face_count += 1
        if face_count >= max_faces:
            break
    cv2.imshow('Detected Faces', frame)
    cv2.waitKey(1)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
