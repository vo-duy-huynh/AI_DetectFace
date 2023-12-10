import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import os
# Load ảnh, chia loại, định lại cỡ ảnh, rescale ảnh
train_datagen = ImageDataGenerator(rescale=1./255,
                                   #                                    shear_range = 0.2,
                                   #                                    zoom_range = 0.2,
                                   #                                    horizontal_flip = True
                                   )
training_set = train_datagen.flow_from_directory('./Data/Faces/training',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')
# Load ảnh, chia loại, định lại cỡ ảnh, rescale ảnh
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('./Data/Faces/testing',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary',
                                            shuffle=False,
                                            )
training_set.class_indices
# Tạo model
model = Sequential()
# Thêm các layer
model.add(Conv2D(32, kernel_size=3,
                 activation='relu',
                 input_shape=(64, 64, 3),
                 padding='same'))
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
# Thiết lập thông số
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
print('Training...')
# Training
# Huấn luyện
model.fit(x=training_set, validation_data=test_set, epochs=10)
# xoá file h5 cũ
if os.path.exists('modelface.h5'):
    os.remove('modelface.h5')
model.save('modelface.h5')
