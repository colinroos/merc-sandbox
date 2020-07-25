import cv2
import numpy as np
import pandas as pd
import file_utils as util
import os

image_path = '../data/2017/images/KAD17-001_Bx1-5_11.5-25.30m_DxO.jpg'
annotes_path_rot = '../out/test_annotes_rot.csv'
annotes_target = 'data/2017/annotations'
image_target = 'data/2017/images'

if not os.path.isdir('out/test_images/'):
    os.mkdir('../out/test_images/')

if not os.path.isdir('out/test_images_pad/'):
    os.mkdir('../out/test_images_pad/')

annotes_paths = util.findfiles(annotes_target, '.csv')
image_paths = util.findfiles(image_target, '.jpg')

# Constants
TARGET_SHAPE = (2304, 3456, 3)
IMAGE_DIMS = (145, 145)
PIXEL_PER_M = 1.5 / 3225.0

for idx, a in enumerate(annotes_paths):
    # Read annotations
    df = pd.read_csv(a, index_col=0)
    print(df.head())

    # Read image
    img = cv2.imread(image_paths[idx], cv2.IMREAD_COLOR)

    # Resize image if not correct dimensions
    if img.shape[1] != TARGET_SHAPE[1]:
        df.loc[:, ['cx', 'cy', 'width', 'height']] *= TARGET_SHAPE[1] / img.shape[1]
        df[['cx', 'cy', 'width', 'height']] = df[['cx', 'cy', 'width', 'height']].astype('int64')
        img = cv2.resize(img, (TARGET_SHAPE[1], TARGET_SHAPE[0]))

    # Segment each image annotation
    for index, d in df.iterrows():
        # Get rotation matrix and transform image accordingly
        matrix = cv2.getRotationMatrix2D(center=(d.cx, d.cy), angle=-(d.rot * 180/np.pi), scale=1)
        image = cv2.warpAffine(src=img, M=matrix, dsize=(img.shape[1], img.shape[0]))
        x = int(d.cx - d.width/2)  # extract left edge
        y = int(d.cy - d.height/2)  # extract bottom edge
        image = image[y:(y+d.height), x:(x + d.width)]

        # Compute depth offset
        if 'x' in d.ToM:
            depth_offset = float(d.FromM)
        elif 'x' in d.FromM:
            depth_offset = float(d.ToM) - (d.width * PIXEL_PER_M)
        else:
            depth_offset = float(d.FromM)

        # Tile image segment
        x_cut = 0
        while x_cut < d.width:
            tile = image[0:d.height, x_cut:min(x_cut + IMAGE_DIMS[1], d.width)]

            if tile.shape[1] < IMAGE_DIMS[1] * 0.5:
                break

            height_padding = max(IMAGE_DIMS[0] - d.height, 0)
            width_padding = max(IMAGE_DIMS[1] - tile.shape[1], 0)
            padding = ((0, height_padding), (0, width_padding), (0, 0))
            padded_tile = np.pad(tile, padding)

            depth = depth_offset + (x_cut * PIXEL_PER_M)

            # write image to file
            filename = 'out/test_images/' + d.HoleID + f'_{depth:.3f}.jpg'
            cv2.imwrite(filename, padded_tile)
            cv2.putText(padded_tile, f'{depth:.3f}', (0, 20), fontFace=0, fontScale=0.5, color=(0, 200, 0))
            # cv2.imshow('', tile)
            # cv2.waitKey(0)

            scaled_tile = cv2.resize(tile, (IMAGE_DIMS[0], tile.shape[1]))
            padding = ((0, 0), (0, width_padding), (0, 0))
            padded_scaled_tile = np.pad(scaled_tile, padding)

            filename = 'out/test_images_pad/' + d.HoleID + f'_{depth:.3f}.jpg'
            cv2.imwrite(filename, padded_scaled_tile)

            x_cut += IMAGE_DIMS[1]
