from sklearn.model_selection import train_test_split
from util.image_pipeline import create_imageset

hole_ids = ['KAD17_001', 'KAD17_002']

df2 = create_imageset('./data-cache/scaled_image_tiles/', hole_ids)

print(df2.info())

# Split the data in to train and test
X_train, X_test, y_train, y_test = train_test_split(df2['path'],
                                                    df2['y'],
                                                    shuffle=True,
                                                    random_state=42,
                                                    train_size=0.8)

# Dump training and testing data to pickle objects
X_train.to_pickle(f'./data-cache/lstm_data/X_train.pkl')
X_test.to_pickle(f'./data-cache/lstm_data/X_test.pkl')
y_train.to_pickle(f'./data-cache/lstm_data/y_train.pkl')
y_test.to_pickle(f'./data-cache/lstm_data/y_test.pkl')

