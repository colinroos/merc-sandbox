import autokeras as ak
import pandas as pd
import numpy as np
import cv2.cv2 as cv
import tensorflow as tf
from sklearn.model_selection import KFold
from util.image_pipeline import load_images_to_memory

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load the training data
X_train_all = pd.read_pickle('./data-cache/lstm_data/X_train.pkl')
y_train_all = pd.read_pickle('./data-cache/lstm_data/y_train.pkl')

# Build the model search space
input_node = ak.ImageInput()
output_node = ak.ImageBlock()(input_node)
output_node = ak.DenseBlock()(output_node)
output_node = ak.RegressionHead()(output_node)

# Train each fold of validation, using the best model from the previous fold
fold = 0
kf = KFold(5, random_state=42)

for _, train_index in kf.split(X_train_all):
    print(f'Fitting fold {fold}: ')

    # Extract training indices from the fold generator
    X_train = X_train_all.iloc[train_index]
    y_train = y_train_all.iloc[train_index]

    X_images = load_images_to_memory(X_train)

    # Convert label format
    y_train = np.array(y_train, dtype=np.float32)

    if fold == 0:
        classifier = ak.AutoModel(inputs=input_node,
                                  outputs=output_node,
                                  max_trials=10,
                                  overwrite=False,
                                  project_name='auto_model')

        classifier.fit(X_images, y_train, epochs=5, batch_size=4)

        models = classifier.tuner.get_best_models(3)
    else:
        for idx, model in enumerate(models):
            model.fit(X_images, y_train, epochs=5, batch_size=4)

            model.save(f'./model-cache/fold_{fold}_model_{idx}.h5')

    fold += 1


