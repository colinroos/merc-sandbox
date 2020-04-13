import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib as plt
import numpy as np
import cv2
import file_utils as util

randomState = 42
annote_path = 'data/2017/images/KAD17-001_Bx1-5_11.5-25.30m_DxO.xml'
search_path = 'data/2017/images/'

xmlpaths = util.findfiles(search_path, '.xml')

for p in xmlpaths:
    tree = ET.parse(p)
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
    columns = ['HoleID', 'FromM', 'ToM', 'cx', 'cy', 'width', 'height', 'rot']
    df = pd.DataFrame(boxes, columns=columns)

    df.to_csv('out/' + image_name[:-4] + '.csv')
    print(df.head())
