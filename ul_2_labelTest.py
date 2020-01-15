import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib as plt
import numpy as np
import cv2

randomState = 42
annote_path = 'data/2017/images/KAD17-001_Bx1-5_11.5-25.30m_DxO.xml'
image_path = 'data/2017/images/KAD17-001_Bx1-5_11.5-25.30m_DxO.jpg'

tree = ET.parse(annote_path)
root = tree.getroot()

# Extract image name
image_name = root.find('.//filename').text
print(image_name)

# Extract image dimensions
image_size = [int(root.find('.//size/width').text), int(root.find('.//size/height').text)]
print(image_size)

boxes = []
# search for objects and extract attributes
for obj in root.findall('.//object'):
    name = obj.find('name').text
    n = name.split('-')
    _hole = n[1]
    _from = n[-2]
    _to = n[-1]

    for box in obj.findall('.//rotated_box'):
        cx = int(box.find('cx').text)
        cy = int(box.find('cy').text)
        width = int(box.find('width').text)
        height = int(box.find('height').text)
        rot = float(box.find('rot').text)
        boxes.append([_hole, _from, _to, cx, cy, width, height, rot])

    for box in obj.findall('.//bndbox'):
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)
        # boxes.append([_hole, _from, _to, xmin, ymin, xmax, ymax])

# Convert to DataFrame
# columns = ['HoleID', 'FromM', 'ToM', 'xmin', 'ymin', 'xmax', 'ymax']
columns = ['HoleID', 'FromM', 'ToM', 'cx', 'cy', 'width', 'height', 'rot']
df = pd.DataFrame(boxes, columns=columns)

df.to_csv('out/test_annotes_rot.csv')
print(df.head())

# load image
img = cv2.imread(image_path)

# Determine largest bounding box
mbox_x = max(df.xmax - df.xmin)
mbox_y = max(df.ymax - df.ymin)

print((mbox_x, mbox_y))

# crop image to 1st box
d = df.iloc[3]
xmin = d.xmin
ymin = d.ymin
xmax = d.xmax
ymax = d.ymax
name = d.HoleID + ' ' + d.FromM + '-' + d.ToM
img_c = img[ymin:ymax, xmin:xmax, :3]

# Pad to largest
img_p = np.pad(img_c, ((0, mbox_y - (ymax - ymin)), (0, mbox_x - (xmax - xmin)), (0, 0)), mode='constant', constant_values=((0, 0), (0, 0), (0, 0)))

# Display image
# cv2.imshow(name, img_c)
cv2.waitKey(0)
cv2.imshow(name, img_p)
cv2.waitKey(0)
