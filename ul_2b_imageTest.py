import cv2
import numpy as np
import pandas as pd
import file_utils as util

randomState = 42
image_path = 'data/2017/images/KAD17-001_Bx1-5_11.5-25.30m_DxO.jpg'
annotes_path_rot = 'out/test_annotes_rot.csv'
annotes_target = 'data/2017/annotations'
image_target = 'data/2017/images'

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
    cv2.imshow('', img)

    # Resize image if not correct dimensions
    if img.shape[1] != TARGET_SHAPE[1]:
        df.loc[:, ['cx', 'cy', 'width', 'height']] *= TARGET_SHAPE[1] / img.shape[1]
        df[['cx', 'cy', 'width', 'height']] = df[['cx', 'cy', 'width', 'height']].astype('int64')
        cv2.resize(img, (TARGET_SHAPE[0], TARGET_SHAPE[1]))
        # cv2.imshow('', img)
        # cv2.waitKey(0)

    for index, d in df.iterrows():
        # Get rotation matrix and transform image accordingly
        matrix = cv2.getRotationMatrix2D(center=(d.cx, d.cy), angle=-(d.rot * 180/np.pi), scale=1)
        image = cv2.warpAffine(src=img, M=matrix, dsize=(img.shape[1], img.shape[0]))
        # cv2.imshow('', image)
        # cv2.waitKey(0)
        x = int(d.cx - d.width/2) # extract left edge
        y = int(d.cy - d.height/2) # extract bottom edge
        image = image[y:(y+d.height), x:(x + d.width)]
        # cv2.imshow('', image)
        # cv2.waitKey(0)

        # Compute depth offset
        if 'x' in d.ToM:
            depth_offset = float(d.FromM)
        elif 'x' in d.FromM:
            depth_offset = float(d.ToM) - (d.width * PIXEL_PER_M)
        else:
            depth_offset = float(d.FromM)

        # Tile images
        x_cut = 0
        while x_cut < d.width:
            if x_cut + IMAGE_DIMS[1] <= d.width:
                tile = image[0:IMAGE_DIMS[0], x_cut:(x_cut + IMAGE_DIMS[1])]
                padding = ((0, IMAGE_DIMS[0] - d.height), (0, 0), (0, 0))
                constantValues = ((0, 0), (0, 0), (0, 0))
                tile = np.pad(tile, padding, constant_values=constantValues)

                depth = depth_offset + (x_cut * PIXEL_PER_M)

                print('{:.3f}'.format(depth))
                # write image to file
                filename = 'out/test_images/' + d.HoleID + '_{:.3f}.jpg'.format(depth)
                cv2.imwrite(filename, tile)
                cv2.putText(tile, '{:.3f}'.format(depth), (0, 20), fontFace=0, fontScale=0.5, color=(0, 200, 0))
                cv2.imshow('', tile)
                cv2.waitKey(0)

            x_cut += IMAGE_DIMS[1]
