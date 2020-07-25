import os
from glob import glob
import xml.etree.ElementTree as ET
import pandas as pd


def findfiles(target, extension='.csv'):
    """Searches a specified directory and locates all absolute file paths.

        Arguments:
            - target    String path to the target directory to be searched
            - extension String of the format '.jpg' to filter by
        Returns:
            - list of file paths (iterable)
    """
    data = []

    for path, subFolders, files in os.walk(target):
        files.sort()
        files = [file for file in files if file.endswith(extension)]
        for file in files:
            p = os.path.join(os.path.abspath(path), file)
            data.append(p)

    data.sort()
    return data


def find_files_glob(target, extension='*.csv', recursive=True):
    """
    Finds a specified type of file using the glob method.
    :param target:
    :param extension:
    :param recursive:
    :return:
    """
    data = []
    files = glob(target + extension, recursive=recursive)
    files.sort()
    for file in files:
        data.append(file)
        # print(file)


    return data


def xml_to_df(path_to_xml):
    """
    Reads an XML file and converts the bounding box data to a dataframe for ease of use.
    :param path_to_xml: Path to XML file.
    :type path_to_xml: String
    :return:
    """
    tree = ET.parse(path_to_xml)
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

        # Process all the rotated box types
        for box in obj.findall('.//rotated_box'):
            cx = int(box.find('cx').text)
            cy = int(box.find('cy').text)
            width = int(box.find('width').text)
            height = int(box.find('height').text)
            rot = float(box.find('rot').text)
            boxes.append([_hole, _from, _to, cx, cy, width, height, rot])

        # Process at the bounding box types
        # for box in obj.findall('.//bndbox'):
        #     xmin = int(box.find('xmin').text)
        #     ymin = int(box.find('ymin').text)
        #     xmax = int(box.find('xmax').text)
        #     ymax = int(box.find('ymax').text)
        #     boxes.append([_hole, _from, _to, xmin, ymin, xmax, ymax])

    # Convert to DataFrame
    columns = ['HoleID', 'FromM', 'ToM', 'cx', 'cy', 'width', 'height', 'rot']
    df = pd.DataFrame(boxes, columns=columns)

    return df
