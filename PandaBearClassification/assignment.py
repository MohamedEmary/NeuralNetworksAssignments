# Import required libraries
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split


# To ignore the warning which caused the model not to be saved
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


# Define directory paths and categories
DATADIR = 'PandasBears'
CATEGORIES = ["Panda", "Bear"]

# Loop over the categories
for category in CATEGORIES:
    # Create the path to the category
    path = os.path.join(DATADIR, category)
    i = 0
    # Loop over the images in the category directory
    for img in os.listdir(path):
        i += 1
        # Read the images in grayscale mode
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        # # Display the images
        # plt.imshow(img_array, cmap='gray')
        # plt.show()
        # Break the loop after displaying 10 images
        if i == 10:
            break
    # Print the images array and its shape
    print(img_array)
    print(img_array.shape)

# Resize the images to the specified size
IMG_SIZE = 200
# # display the resized images to see if we can identify the panda or bear or not
# new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
# plt.imshow(new_array, cmap='gray')
# plt.show()


# Define a function to create the training data
training_data = []


def create_training_data():
    # Loop over the categories
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        # Get the classification  (0 or a 1). 0=Panda 1=Bear
        class_num = CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):
            try:
                # Read the images in grayscale mode
                img_array = cv2.imread(os.path.join(
                    path, img), cv2.IMREAD_GRAYSCALE)
                # Resize the images to the specified size
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                # Append the resized images and its labels to the training data list
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


# Call the function to create the training data
create_training_data()

# # print training_data to see the structure of it
# print(training_data[:3])

# Print the length of the training data
print("Length of the training data: ", len(training_data))

# Shuffle the training data randomly
random.shuffle(training_data)
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])

# Extract the features and labels from the training data and store them in images and labels variables
images = []
labels = []
for features, label in training_data:
    images.append(features)
    labels.append(label)

# Reshape the features array to be compatible with the input shape of the model
images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE)
labels = np.array(labels)

# Save the features and labels in pickle files
pickle_out = open("images.pickle", "wb")
pickle.dump(images, pickle_out)
pickle_out.close()

pickle_out = open("labels.pickle", "wb")
pickle.dump(labels, pickle_out)
pickle_out.close()

# Load the features and labels from the pickle files
pickle_in = open("images.pickle", "rb")
images = pickle.load(pickle_in)

pickle_in = open("labels.pickle", "rb")
labels = pickle.load(pickle_in)


images_train, images_test, labels_train, labels_test = train_test_split(
    images, labels, test_size=0.2, random_state=42)

images_train, images_test = images_train / 255.0, images_test / 255.0

print(
    f"Training data shape: {images_train.shape}, {labels_train.shape}, Test data shape: {images_test.shape}, {labels_test.shape}")


# Define the model architecture
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE)),
    keras.layers.Dense(1024, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

# Compile the model with an optimizer, loss function, and metric
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model on the training data for a specified number of epochs
model.fit(images_train, labels_train, epochs=10)

# Evaluate the model on the training data and print the accuracy
test_loss, test_acc = model.evaluate(images_test, labels_test)

print("Model Summary: ")
model.summary()

print(f"Test Accuracy is : {test_acc} Test Loss is: {test_loss}")

model.save('pandas_bears.model')
# model.load('pandas_bears.model')

'''
output = model.predict(x_test)
y_pred = np.argmax(output, axis=1)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_pred, y_test))
print(confusion_matrix(y_test, y_pred))
'''