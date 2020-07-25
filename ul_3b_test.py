import autokeras as ak
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from util.image_pipeline import load_images_to_memory

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load the testing data
X_test = pd.read_pickle('./data-cache/lstm_data/X_test.pkl')
y_test = pd.read_pickle('./data-cache/lstm_data/y_test.pkl')

X_images = load_images_to_memory(X_test)

# Convert lists to arrays
y_test = np.array(y_test, dtype=np.float32)

# Build the model search space
classifier = tf.keras.models.load_model('./model-cache/fold_0_model.h5', custom_objects=ak.CUSTOM_OBJECTS)
print(classifier.summary())

y_pred = classifier.predict(X_images, batch_size=4)

mse = mean_squared_error(y_test, y_pred)

print(f'MSE: {mse}')


