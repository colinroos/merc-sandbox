import pandas as pd
import numpy as np
import file_utils as util
from sklearn.model_selection import train_test_split
from tqdm import tqdm

image_paths = util.findfiles('out/test_images', '.jpg')

df = pd.DataFrame()

df['image_path'] = image_paths

labels = pd.read_pickle('out/merged_master.pkl')
labels['shifted'] = labels['From_m'].shift(-1).fillna(0)

y = []
p = []
for idx, row in tqdm(df.iterrows()):
    s = row['image_path'][row['image_path'].find('KAD17'):-4].split('_')
    hole = s[0] + '-' + s[1]
    depth = float(s[2])

    lab = labels.loc[(labels['From_m'] < depth) & (depth <= labels['shifted']) & (labels['Hole'] == hole)]
    y.append(np.asarray(lab['Alt_Group']))
    p.append(row)

    if idx > 100:
        break

df2 = pd.DataFrame()
df2['path'] = p
df2['y'] = pd.Categorical(y)

print(df2.head())
