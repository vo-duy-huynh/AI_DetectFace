from sklearn.preprocessing import LabelBinarizer
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import os
import cv2
data = []
label = []
path_folder = 'Data/Faces/training'
list_folder = os.listdir(path_folder)
for folder in list_folder:
    list_file = os.listdir(path_folder + '/' + folder)
    sttfolder = list_folder.index(folder)
    for file in list_file:
        filename = path_folder + '/' + folder + '/' + file
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(src=img, dsize=(100, 100))
        img = np.array(img)
        data.append(img)
        label.append(sttfolder)
data1 = np.array(data)
label = np.array(label)
print(data1.shape)
print(label.shape)
print(label)
data1 = data1.reshape((len(label), 100, 100, 3))
x_train = data1 / 255
le = LabelEncoder()
trainY = le.fit_transform(label)
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(len(list_folder), activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print('Training...')
print(trainY)
print(x_train.shape)
model.fit(x_train, trainY, epochs=5, batch_size=5)
model.save('modelface.h5')
