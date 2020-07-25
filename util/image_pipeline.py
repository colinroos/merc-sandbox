import cv2
import numpy as np
import pandas as pd
from util.files import xml_to_df, find_files_glob
import os
from tqdm import tqdm


# Constants
TARGET_SHAPE = (2304, 3456, 3)
IMAGE_DIMS = (145, 145)
PIXEL_PER_M = 1.5 / 3225.0


def tile_image_from_annotation(annotation_path, image_path):
    """
    Uses a annotation file to extract a region of interest and tiles it to a target size. Image tiles are
    saved with their depth in the filename for retrieval later.
    :param annotation_path: The path the XML file for the image
    :type annotation_path: String
    :param image_path: The path the respective image file.
    :type image_path: String
    :return: Null
    """

    # Make sure the output folders exist
    if not os.path.isdir('./data-cache/padded_image_tiles/'):
        os.mkdir('./data-cache/padded_image_tiles/')

    if not os.path.isdir('./data-cache/scaled_image_tiles/'):
        os.mkdir('./data-cache/scaled_image_tiles/')

    # Read annotations
    df = xml_to_df(annotation_path)
    print(df.head())

    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

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
            # Slice the image to create the tile
            tile = image[0:d.height, x_cut:min(x_cut + IMAGE_DIMS[1], d.width)]

            # Skip if we get a tile that is too small
            if tile.shape[1] < IMAGE_DIMS[1] * 0.5:
                break

            # Calculate the required padding
            height_padding = max(IMAGE_DIMS[0] - d.height, 0)
            width_padding = max(IMAGE_DIMS[1] - tile.shape[1], 0)
            padding = ((0, height_padding), (0, width_padding), (0, 0))

            # Pad the tile with zeros (black) so we have a constant input size to the network
            padded_tile = np.pad(tile, padding)

            # Calculate the depth offset of the tile's left edge to label the tile
            depth = depth_offset + (x_cut * PIXEL_PER_M)

            # Write tile to file
            filename = './data-cache/padded_image_tiles/' + d.HoleID + f'_{depth:.3f}.jpg'
            cv2.imwrite(filename, padded_tile)

            # Scales the tile vertically to the target size instead of padding it
            scaled_tile = cv2.resize(tile, (IMAGE_DIMS[0], tile.shape[1]))

            # Calculate the horizontal padding required
            padding = ((0, 0), (0, width_padding), (0, 0))

            # Pad the tile to match target dimensions
            padded_scaled_tile = np.pad(scaled_tile, padding)

            # Write the scaled images to file
            filename = './data-cache/scaled_image_tiles/' + d.HoleID + f'_{depth:.3f}.jpg'
            cv2.imwrite(filename, padded_scaled_tile)

            # Increment the image slice index
            x_cut += IMAGE_DIMS[1]


def extract_depth(row):
    s = row[row.find('KAD17'):-4].split('_')
    hole = s[0] + '-' + s[1]
    depth = float(s[2])

    return hole, depth


def create_imageset(path_to_images, hole_id):
    """
    Creates an array of image path sets and their associated labels.
    :param path_to_images:
    :return:
    """
    # Correct argment type
    if type(hole_id) != list:
        hole_id = [hole_id]

    n_images = 3
    y = []
    p = []

    for hole in hole_id:
        # Search for all the image files in a specified folder
        image_paths = find_files_glob(path_to_images, '*' + hole + '*.jpg')

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

        print(f'Processing hole {hole}...')

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

    return df2


def load_images_to_memory(image_df):
    """
    Reads a list if image files in a sets and loads the images to memory as an np.array
    :param image_df:
    :return: the loaded images as an np.array
    """

    # Loop through each row and load the list of images for that row
    X_images = []
    for row in image_df:

        # Load all the images in the list.
        temp_img = []
        for i, path in enumerate(row):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img.shape != (145, 145, 3):
                img = cv2.resize(img, (145, 145))

            temp_img.append(img)

        X_images.append(np.hstack(temp_img))

    return np.array(X_images)
