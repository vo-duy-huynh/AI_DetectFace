import numpy as np
import cv2
import os

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

path_folder = 'Images'
list_folder = os.listdir(path_folder)

for folder in list_folder:
    list_file = os.listdir(path_folder + '/' + folder)

    # Create a folder for the person in the 'dataset' directory
    person_folder = 'dataset/' + folder
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)

    for file in list_file:
        filename = path_folder + '/' + folder + '/' + file
        frame = cv2.imread(filename)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Extract the face
            face = gray[y:y+h, x:x+w]

            # Save the face in the person's folder
            cv2.imwrite(person_folder + '/' + file, face)

            # Draw a rectangle on the original image (optional)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Note: If you want to show the images with rectangles, uncomment the following line
# cv2.imshow('Detected Faces', frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
