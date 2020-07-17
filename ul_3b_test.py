import autokeras as ak
import pandas as pd
import numpy as np
import cv2.cv2 as cv
import tensorflow as tf
from sklearn.metrics import mean_squared_error

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load the training data
X_test = pd.read_pickle('./data-cache/lstm_data/X_test.pkl')
y_test = pd.read_pickle('./data-cache/lstm_data/y_test.pkl')

# Extract the number of images from the length of image lists
n_images = len(X_test.iloc[0])

# Loop through each row and load the list of images for that row
X_images = [[] for i in range(n_images)]
for row in X_test:

    # Load all the images in the list
    for i, path in enumerate(row):
        img = cv.imread(path, cv.IMREAD_COLOR)
        if img.shape != (145, 145, 3):
            img = cv.resize(img, (145, 145))

        # img = np.expand_dims(img, axis=0)
        X_images[i].append(img)

        if img.shape != (145, 145, 3):
            pass

# Convert images to a stacked array, convert datatype
# TODO test without float conversion
temp_images = []
for image in X_images:
    temp_images.append(np.stack(image, axis=0).astype(np.float32))

X_images = temp_images
# Convert label
y_test = np.array(y_test, dtype=np.float32)

# Build the model search space
classifier = tf.keras.models.load_model('./model-cache/model.h5', custom_objects=ak.CUSTOM_OBJECTS)

y_pred = classifier.predict(X_images, batch_size=4)

mse = mean_squared_error(y_test, y_pred)

print(f'MSE: {mse}')
