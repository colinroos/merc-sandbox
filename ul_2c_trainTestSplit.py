import pandas as pd
import numpy as np
import file_utils as util
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def extract_depth(row):
    s = row[row.find('KAD17'):-4].split('_')
    hole = s[0] + '-' + s[1]
    depth = float(s[2])

    return hole, depth


# Search for all the image files in a specified folder
image_paths = util.findfiles('out/test_images', '.jpg')

# Store the image paths in a new DataFrame for ease-of-use
df = pd.DataFrame()
df['image_path'] = image_paths

# Extract the depth so that it can be used to sort the DataFrame
df['depth'] = df['image_path'].apply(lambda x: extract_depth(x)[1])

# Sort the images paths by depth instead of alphanumeric
df.sort_values(by='depth', inplace=True)
df.reset_index(inplace=True)

# Load the labels object to a DataFrame
labels = pd.read_pickle('out/merged_master.pkl')
labels['shifted'] = labels['From_m'].shift(-1).fillna(0)

n_images = 3
y = []
p = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    # Start condition
    if idx <= int(n_images / 2):
        continue

    # End condition
    if idx >= len(df) - int(n_images / 2):
        break

    # Extract the depth and hole information from the image path
    hole, depth = extract_depth(row['image_path'])

    # Loop up the appropriate label for the image, given depth and hole ID
    lab = labels.loc[(labels['From_m'] < depth) & (depth <= labels['shifted']) & (labels['Hole'] == hole)]

    # If there is no label found, skip this one
    if len(lab) == 0:
        continue

    y.append(lab['RQD_m'].values[0])

    list_of_images = []
    start_idx = idx - int(n_images / 2)
    # Iterate through the images that are around this label
    for i in range(n_images):
        list_of_images.append(df['image_path'].iloc[start_idx + i])

    # Store the list of image paths for this label
    p.append(list_of_images)

# Assemble features and targets into a DataFrame
df2 = pd.DataFrame()
df2['path'] = p
df2['y'] = y

print(df2.info())

# Split the data in to train and test
X_train, X_test, y_train, y_test = train_test_split(df2['path'],
                                                    df2['y'],
                                                    shuffle=True,
                                                    random_state=42,
                                                    train_size=0.8)

# Dump training and testing data to pickle objects
X_train.to_pickle('./data-cache/lstm_data/X_train.pkl')
X_test.to_pickle('./data-cache/lstm_data/X_test.pkl')
y_train.to_pickle('./data-cache/lstm_data/y_train.pkl')
y_test.to_pickle('./data-cache/lstm_data/y_test.pkl')

