from keras.preprocessing import image
from keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np

# Load the model
model = load_model('modelface.h5')

# Print the summary of the model
print(model.summary())

# Define class names (replace with actual class names)
class_names = ["Person1", "Person2", "Person3", "Person4", "Person5"]

# Load and preprocess the test image
test_image = image.load_img(
    './Data/Faces/testing/minhduc/minhduc.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
plt.imshow(test_image/255)  # Display the image
test_image = np.expand_dims(test_image, axis=0)

# Make predictions
result = model.predict(test_image)

# Print detailed predictions
print("Raw predictions:", result)
predicted_class = np.argmax(result)
confidence = result[0][predicted_class]

print(f"Predicted class index: {predicted_class}")
print(f"Predicted class: {class_names[predicted_class]}")
print(f"Confidence: {confidence:.2%}")
