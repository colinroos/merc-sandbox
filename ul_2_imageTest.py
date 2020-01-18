import cv2
import numpy as np
import pandas as pd


randomState = 42
image_path = 'data/2017/images/KAD17-001_Bx1-5_11.5-25.30m_DxO.jpg'
annotes_path_rot = 'out/test_annotes_rot.csv'

# read annotations
df = pd.read_csv(annotes_path_rot)
print(df.head())
# read image
img = cv2.imread(image_path)
mbox_x = max(df.width)
mbox_y = max(df.height)

for index, d in df.iterrows():
    print(d.cx)
    # get rotation matrix and transform image accordingly
    matrix = cv2.getRotationMatrix2D(center=(d.cx, d.cy), angle=-(d.rot * 180/np.pi), scale=1)
    image = cv2.warpAffine(src=img, M=matrix, dsize=(img.shape[1], img.shape[0]))
    x = int(d.cx - d.width/2) # extract left edge
    y = int(d.cy - d.height/2) # extract bottom edge
    image = image[y:(y+d.height), x:(x + d.width)]
    padding = ((0, mbox_y - d.height), (0, mbox_x - d.width), (0, 0))
    constantValues = ((0, 0), (0, 0), (0, 0))
    image = np.pad(image, padding, constant_values=constantValues)

    # write image to file
    filename = 'out/test_images/' + d.HoleID + '_' + str(d.FromM) + '_' + str(d.ToM) + '.jpg'
    cv2.imwrite(filename, image)
