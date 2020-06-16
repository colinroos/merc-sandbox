import numpy as np
import cv2
from sympy.geometry import Point, Line, intersection
from PyQt5 import QtGui
from scipy.signal import find_peaks


def auto_canny(image, sigma=0.33):
    v = np.median(image)

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    return cv2.Canny(image, lower, upper)


def blur(image, kernel):
    """
    Applies a Gaussian Blur to an image using the specified kernel.
    :param image: Image to be blurred, array-like object
    :param kernel: n x n kernel size
    :return: Blurred image, array-like object
    """
    img = cv2.GaussianBlur(image.copy(), (kernel, kernel), 0)
    return img


def transform_points(points, matrix):
    """
    Transforms a list of points by transformation matrix.
    :param points: list of points to be transformed [[x1, y1], [x2, y2]]
    :param matrix: transformation matrix, 3x3
    :return: transformed points
    """
    nd_points = np.float32(np.array(points)).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(nd_points, matrix)


def crop_to_roi(image, roi):
    """
    Crops an image to the specified ROI
    :param image:
    :param roi: (x1, y1) , (x2, y2)
    :return:
    """
    if len(image.shape) == 3:
        img = image[:1000, :1000, :]
    else:
        img = image[roi[0][0]:roi[1][0], roi[0][1]:roi[1][1]]
    return img


def find_edge(image,
              roi=None,  # region of interest, 2 points
              mode=0,  # 0 horizontal, 1 vertical
              right_start=False,  # scan right to left if true
              threshold=100,  # binary rising/falling edge value
              detectors=None,  # list of slices to scan
              start_pixel=0,  # start scanning pixel
              end_pixel=-1,  # end scanning pixel
              max_blob=5,  # minimum length of acceptable edge
              noise_filter_length=0):
    """
    finds an edge
    :param std_size:
    :param std_edge_mode:
    :param noise_filter_length:
    :param right_start:
    :param end_pixel:
    :param max_blob:
    :param nest:
    :param start_pixel:
    :param show_plot:
    :param image:
    :param roi:
    :param mode:
    :param threshold:
    :param detectors:
    :return:
    """
    try:
        img = image.copy()
        if roi is not None:
            img = crop_to_roi(img, roi)

        # Check if the image is color and remove color channels for transpose
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Transpose the image if horizontal mode
        if mode == 1:
            img = img.T

        # Populate detectors object if no detector specified
        if detectors is None:
            w, h = img.shape[:2]
            mid = int(h / 2)
            offset = int(mid * 0.2)
            detectors = [mid - offset, mid, mid + offset]

        if type(detectors) is not list:
            detectors = [detectors]

        rising_edges = []
        falling_edges = []
        rising_blobs = []
        falling_blobs = []

        for detector in detectors:
            img_slice = img[:, detector]

            temp_img = image.copy()
            temp_img[:, detector] = (0, 255, 0)
            cv2.imshow('', temp_img)
            # cv2.waitKey(0)

            # If start from the right, flip the slice
            if right_start:
                img_slice = img_slice[::-1]

            # Loop through sliced image and find rising and falling edges
            cropped_img_slice = img_slice[start_pixel:end_pixel - noise_filter_length]

            slope = np.gradient(cropped_img_slice)
            peaks, params = find_peaks(slope, distance=10)
            peaks = [peak + start_pixel for peak in peaks]
            rising_blobs.extend(peaks)

        else:
            if right_start:
                rising_edges = [len(img_slice) - edge for edge in rising_edges]
                falling_edges = [len(img_slice) - edge for edge in falling_edges]
                rising_blobs = [len(img_slice) - edge for edge in rising_blobs]
                falling_blobs = [len(img_slice) - edge for edge in falling_blobs]

        return rising_edges, falling_edges, rising_blobs, falling_blobs
    except Exception as e:
        print('error in find_edge')
        print(e)
        return 0, 0, 0, 0


def get_transform(transform):
    """
    Extracts the translation and rotation information from a transformation matrix
    :param transform: 3x3 transformation matrix
    :return: x translation, y translation, rotation angle (deg)
    """
    theta0 = np.degrees(np.arctan2(-transform[0, 1], transform[0, 0]))
    theta1 = np.degrees(np.arctan2(transform[1, 0], transform[1, 1]))
    theta = np.average([theta1, theta0])

    t_x = transform[0, 2]
    t_y = transform[1, 2]

    return t_x, t_y, theta
