import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Đường dẫn đến thư mục chứa ảnh
data_dir = r'D:\NCKH\AI\Images\testcode'

# Đọc và chuyển đổi ảnh thành mảng numpy


def read_and_process_images(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            # Thay đổi kích thước thành kích thước ảnh mong muốn
            img = cv2.resize(img, (224, 224))
            img = img / 255.0  # Chuẩn hóa giá trị pixel về khoảng [0, 1]
            images.append(img)

            # Đặt nhãn dựa trên tên của file hoặc cách khác
            if "lee" in filename:
                labels.append(0)
            elif "cat_bw" in filename:
                labels.append(1)

    return np.array(images), np.array(labels)


# Đọc và tiền xử lý dữ liệu
images, labels = read_and_process_images(data_dir)

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42)

# Xây dựng mô hình
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(224, 224, 3)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# Đào tạo mô hình
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Đánh giá mô hình
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Dự đoán
new_images = np.array(
    [cv2.resize(cv2.imread(r'D:\NCKH\AI\Images\testcode\lee.jpg'), (224, 224)) / 255.0])
predictions = model.predict(new_images)
print(f'Predictions: {predictions}')
