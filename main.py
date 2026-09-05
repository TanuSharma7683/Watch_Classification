from preprocessing import prepare_dataset
from model import train_model
import cv2
import numpy as np
import os


print("==============================================")
print("     WATCH DETECTION USING MACHINE LEARNING")
print("==============================================")


# Load dataset
print("\nLoading images...")

X, Y = prepare_dataset()

print("Total images:", len(X))

print("With Watch images:", np.sum(Y == 1))
print("Without Watch images:", np.sum(Y == 0))


# Train model
print("\nTraining SVM model...")

model = train_model(X, Y)

print("\nTraining completed successfully!")


# Prediction
print("\n==============================================")
print("             IMAGE PREDICTION")
print("==============================================")


image_path = input(
    "\nEnter the path of an image to predict: "
)

# Remove quotation marks if user pasted a Windows path
image_path = image_path.strip('"')


image = cv2.imread(image_path)


if image is None:

    print("\nInvalid image path!")

else:

    # Resize image
    image = cv2.resize(image, (64, 64))

    # Convert image into 1D array
    image = image.flatten()

    # Convert into NumPy array
    image = np.array([image])

    # Prediction
    prediction = model.predict(image)

    print("\n----------------------------------------------")

    if prediction[0] == 1:
        print("Prediction: WITH WATCH")
    else:
        print("Prediction: WITHOUT WATCH")

    print("----------------------------------------------")

print("\nProgram completed.")
