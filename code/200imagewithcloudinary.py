import cv2
import os
import time
import cloudinary
from cloudinary.uploader import upload
import pyrebase

config = {
    'apiKey': "AIzaSyA3Ytykw495KxTZIbShqwiCCl4lHgztsos",
    'authDomain': "face-test-45fbe.firebaseapp.com",
    'projectId': "face-test-45fbe",
    'databaseURL': 'https://face-test-45fbe-default-rtdb.firebaseio.com/',
    'storageBucket': "face-test-45fbe.appspot.com",
    'messagingSenderId': "47147490746",
    'appId': "1:47147490746:web:7221c1fb3e94a809075183",
    'measurementId': "G-TENMM8MBR4"}
firebase = pyrebase.initialize_app(config)
db = firebase.database()
cloudinary.config(
    cloud_name="dvcwwbrqw",
    api_key="897574184663944",
    api_secret="4o1bk6kX2H-mDE2iTmChYnVUklM"
)
save_folder = input("Enter the folder name: ")
count = [0]
data = []
try:
    datafromdb = db.child('faces').get().val()
    print(datafromdb)
    for i in datafromdb:
        data.append(i)
    datanew = save_folder
    data.append(datanew)
    db.child('faces').set(data)

except:
    db.child('faces').set(save_folder)

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
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_filename = os.path.join(
            'Data/Faces/training', save_folder, f'face_{face_count}.jpg')
        cv2.imwrite(face_filename, face_roi)
        cloudinary_response = upload(
            face_filename,
            folder=f'Data/Faces/training/{save_folder}',
            public_id=f'face_{face_count}'
        )
        face_count += 1
        if face_count >= max_faces:
            break
    cv2.imshow('Detected Faces', frame)
    cv2.waitKey(1)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
