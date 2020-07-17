import autokeras as ak
import pandas as pd
import numpy as np
import cv2.cv2 as cv
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load the training data
X_train = pd.read_pickle('./data-cache/lstm_data/X_train.pkl')
y_train = pd.read_pickle('./data-cache/lstm_data/y_train.pkl')

# Extract the number of images from the length of image lists
n_images = len(X_train.iloc[0])

# Loop through each row and load the list of images for that row
X_images = [[] for i in range(n_images)]
for row in X_train:

    # Load all the images in the list
    for i, path in enumerate(row):
        img = cv.imread(path, cv.IMREAD_COLOR)
        if img.shape != (145, 145, 3):
            img = cv.resize(img, (145, 145))

        # img = np.expand_dims(img, axis=0)
        X_images[i].append(img)

        if img.shape != (145, 145, 3):
            pass

    # Store the list of images as an array
    # X_images.append(np.array(temp_imgs))

temp_images = []
for image in X_images:
    temp_images.append(np.stack(image, axis=0).astype(np.float32))

X_images = temp_images
# Convert lael
y_train = np.array(y_train, dtype=np.float32)

# Build the model search space
input_node1 = ak.ImageInput()
input_node2 = ak.ImageInput()
input_node3 = ak.ImageInput()
output_node1 = ak.ImageBlock(augment=False)(input_node1)
output_node2 = ak.ImageBlock(augment=False)(input_node2)
output_node3 = ak.ImageBlock(augment=False)(input_node3)
output_node = ak.Merge()([output_node1, output_node2, output_node3])
output_node = ak.DenseBlock()(output_node)
output_node = ak.RegressionHead()(output_node)

classifier = ak.AutoModel(inputs=[input_node1, input_node2, input_node3], outputs=output_node, max_trials=2, overwrite=True)

classifier.fit(X_images, y_train, epochs=5, batch_size=4)

model = classifier.export_model()

model.save('./model-cache/model.h5')
