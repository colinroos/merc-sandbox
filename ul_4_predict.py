import autokeras as ak
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from util.image_pipeline import create_imageset, load_images_to_memory

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

df = create_imageset('./data-cache/scaled_image_tiles/', 'KAD17_003')

X_images = load_images_to_memory(df['path'])

for model in range(3):
    # Load the model
    classifier = tf.keras.models.load_model(f'./model-cache/fold_4_model_{model}.h5', custom_objects=ak.CUSTOM_OBJECTS)

    y_pred = classifier.predict(X_images, batch_size=4, verbose=True)

    df_out = pd.DataFrame()
    df_out['y'] = df['y']
    df_out['y_pred'] = y_pred

    mse = mean_squared_error(df['y'], y_pred)
    print(f'MSE: {mse}')

    df_out.to_csv(f'./output-cache/RQD_prediction_{model}.csv')


