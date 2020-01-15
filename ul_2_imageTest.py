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

# pull first record
d = df.iloc[0]
print(d.cx)
# get rotation matrix
matrix = cv2.getRotationMatrix2D(center=(d.cx, d.cy), angle=-(d.rot * 180/np.pi), scale=1)

image = cv2.warpAffine(src=img, M=matrix, dsize=(img.shape[1], img.shape[0]))
x1 = int(d.cx - d.width/2)
x2 = int(d.cx + d.width/2)
print((x1,x2))
y = int(d.cy - d.height/2)
image = image[y:(y+d.height), x1:x2]
print(image.shape)

cv2.imshow('', image)
cv2.waitKey(0)
