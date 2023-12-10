import os
import cv2
import numpy as np

path_folder = 'Data/Faces/training/huynhpro'
for file in os.listdir(path_folder):
    filename = path_folder + '/' + file
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(src=img, dsize=(100, 100))
    img = np.array(img)
    # lưu ảnh
    cv2.imwrite(filename, img)
