import os
import cv2
import numpy as np


path_folder = 'Data/Faces/training'
list_folder = os.listdir(path_folder)
i = 1
# đổi tên file trong folder
for folder in list_folder:
    list_file = os.listdir(path_folder + '/' + folder)
    for file in list_file:
        # thứ tự của folder
        filename = path_folder + '/' + folder + '/' + file
        os.rename(filename, path_folder + '/' +
                  folder + '/' + folder + '-' + str(i) + '.jpg')
        if i == 200:
            i = 0
        i += 1
