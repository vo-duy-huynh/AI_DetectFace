import cv2
import os
import time
import cloudinary
from cloudinary.uploader import upload
import pyrebase
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
import os
import cv2
import cloudinary
from cloudinary.uploader import upload
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
data = []
try:
    datafromdb = db.child('faces').get().val()
except:
    pass
list_folder = datafromdb
data = []
labels = []


def train_model_on_folders(model, folders_to_train, data, labels):
    print('Training on folders:', folders_to_train)
    le = LabelEncoder()
    labels_in_folders = le.fit_transform(list_folder)
    le.classes_ = list_folder

    for folder in folders_to_train:
        folder_path = os.path.join('Data/Faces/training', folder)
        list_files = os.listdir(folder_path)
        label = labels_in_folders[list_folder.index(folder)]
        for file in list_files:
            filename = os.path.join(folder_path, file)
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(src=img, dsize=(100, 100))
            img = np.array(img)
            data.append(img)
            labels.append(label)


if os.path.isfile('modelface.h5'):
    face_recognition_model = tf.keras.models.load_model('modelface.h5')
    print(face_recognition_model.summary())
    labels_in_model = face_recognition_model.layers[-1].get_weights()[1]
    print("Labels in the model:", labels_in_model)
    path_folder = 'Data/Faces/training'
    le = LabelEncoder()
    labels_in_folders = le.fit_transform(list_folder)
    print("Labels in the folders:", labels_in_folders)
    missing_labels = set(labels_in_folders) - set(labels_in_model)
    print("Labels missing in the model:", missing_labels)
    if len(missing_labels) > 0:
        print('Các thư mục chưa được train:')
        for folder in missing_labels:
            print(list_folder[folder])
    print('Do you want to train again?')
    print('1. Yes')
    print('2. No')
    print('3. Custom folders')
    choice = int(input('Your choice: '))

    if choice == 2:
        train_model_on_folders(face_recognition_model,
                               missing_labels, data, labels)
    elif choice == 1:
        for folder in list_folder:
            train_model_on_folders(face_recognition_model, [
                                   folder], data, labels)
    elif choice == 3:
        print('Danh sách các folder:')
        for i, folder in enumerate(list_folder):
            print(f'{i}. {folder}')
        custom_folders = input(
            'Nhập index của các folder cần train (cách nhau bởi dấu phẩy): ')
        custom_folders = custom_folders.split(',')
        custom_folders = [int(x) for x in custom_folders]
        custom_folders = [list_folder[x] for x in custom_folders]
        train_model_on_folders(face_recognition_model,
                               custom_folders, data, labels)
    else:
        print('Invalid choice. Exiting...')
        exit()
    data = np.array(data)
    labels = np.array(labels)
    data = data.reshape((len(labels), 100, 100, 3))
    x_train = data / 255
    le = LabelEncoder()
    trainY = le.fit_transform(labels)
    face_recognition_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                                   metrics=['accuracy'])
    print('Training...')
    print(trainY)
    print(x_train.shape)
    face_recognition_model.fit(x_train, trainY, epochs=10, batch_size=5)
    face_recognition_model.save('modelface.h5')
