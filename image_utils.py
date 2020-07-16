import numpy as np
import cv2.cv2 as cv
from sympy.geometry import Point, Line, intersection
from PyQt5 import QtGui
from scipy.signal import find_peaks

# Suppress future warnings on import
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from elbow import KElbowVisualizer
from sklearn.cluster import KMeans, MiniBatchKMeans


def auto_canny(image, sigma=0.33):
    v = np.median(image)

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    return cv.Canny(image, lower, upper)


def blur(image, kernel):
    """
    Applies a Gaussian Blur to an image using the specified kernel.
    :param image: Image to be blurred, array-like object
    :param kernel: n x n kernel size
    :return: Blurred image, array-like object
    """
    img = cv.GaussianBlur(image.copy(), (kernel, kernel), 0)
    return img


def transform_points(points, matrix):
    """
    Transforms a list of points by transformation matrix.
    :param points: list of points to be transformed [[x1, y1], [x2, y2]]
    :param matrix: transformation matrix, 3x3
    :return: transformed points
    """
    nd_points = np.float32(np.array(points)).reshape(-1, 1, 2)
    return cv.perspectiveTransform(nd_points, matrix)


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
              noise_filter_length=0,
              std_edge_mode=False,
              std_size=10):  # A is 0, B is 1
    """
    finds an edge
    :param std_size:
    :param std_edge_mode:
    :param noise_filter_length:
    :param right_start:
    :param end_pixel:
    :param max_blob:
    :param start_pixel:
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
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

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

            # temp_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            # temp_img[detector, :] = (0, 255, 0)
            # cv.imshow('slice', temp_img)
            # cv.waitKey(10)

            # If start from the right, flip the slice
            if right_start:
                img_slice = img_slice[::-1]

            standard_deviations = []
            # Loop through sliced image and find rising and falling edges
            cropped_img_slice = img_slice[start_pixel:end_pixel - noise_filter_length]
            for idx, pixel in enumerate(cropped_img_slice):

                if std_edge_mode:
                    if std_size < idx < (len(cropped_img_slice) - std_size):
                        front_avg = np.mean(img_slice[idx + 1: idx + std_size])
                        back_avg = np.mean(img_slice[idx - std_size: idx - 1])
                        front_std = np.std(img_slice[idx + 1: idx + std_size])
                        back_std = np.std(img_slice[idx - std_size: idx - 1])

                        # Rising Edge
                        standard_deviations.append(front_avg - back_avg)

                else:
                    if idx >= len(cropped_img_slice) - 1:
                        if len(rising_edges) == 0 and len(falling_edges) > 0:
                            falling_blobs.append(falling_edges[-1])
                        elif len(falling_edges) == 0 and len(rising_edges) > 0:
                            rising_blobs.append(rising_edges[-1])
                        break

                    current_idx = idx + start_pixel
                    # Get next pixel and to compare to current pixel
                    next_pixel = img_slice[current_idx + 1]

                    # Rising edge
                    if pixel < threshold <= next_pixel:

                        # Append rising edge
                        if len(rising_edges) == 0 or len(falling_edges) == 0:
                            rising_edges.append(current_idx + 1)
                        elif (current_idx - rising_edges[-1]) > noise_filter_length and \
                                (current_idx - falling_edges[-1]) > noise_filter_length:
                            rising_edges.append(current_idx + 1)

                        # Calculate length of edge and append to blobs if at least size of max_blob
                        if len(falling_edges) > 0 and len(rising_edges) > 0:
                            falling_length = abs(falling_edges[-1] - rising_edges[-1])

                            if falling_length >= max_blob and falling_edges[-1] > 0:
                                falling_blobs.append(falling_edges[-1])

                    # Falling edge
                    if pixel > threshold >= next_pixel:

                        # Append rising edge
                        if len(falling_edges) == 0 or len(rising_edges) == 0:
                            falling_edges.append(current_idx + 1)
                        elif (current_idx - falling_edges[-1]) > noise_filter_length and \
                                (current_idx - rising_edges[-1]) > noise_filter_length:
                            falling_edges.append(current_idx + 1)

                        # Append falling edge
                        falling_edges.append(idx + start_pixel)

                        # Calculate length of edge and append to blobs if at least size of max_blob
                        if len(falling_edges) > 0 and len(rising_edges) > 0:
                            rising_length = abs(rising_edges[-1] - falling_edges[-1])

                            if rising_length >= max_blob and rising_edges[-1] > 0:
                                rising_blobs.append(rising_edges[-1])

            if std_edge_mode:
                local_maxima, params = find_peaks(standard_deviations, distance=10, height=10)
                rising_blobs.extend(local_maxima)

        if len(rising_blobs) == 0 and len(rising_edges) == 1:
            rising_blobs.append(rising_edges[0])

        if len(falling_blobs) == 0 and len(falling_edges) == 1:
            falling_blobs.append(falling_edges[0])

        if right_start:
            rising_edges = [len(img_slice) - edge for edge in rising_edges]
            falling_edges = [len(img_slice) - edge for edge in falling_edges]
            rising_blobs = [len(img_slice) - edge for edge in rising_blobs]
            falling_blobs = [len(img_slice) - edge for edge in falling_blobs]

        return rising_edges, falling_edges, rising_blobs, falling_blobs
    except Exception as e:
        print('error in find_edge')
        print(e)


def rotate_image(image, angle, center=None):
    """
    Rotates an image about its center by an angle
    :param image: image to be rotated
    :param angle: integer angle to rotate by in degrees
    :param center: rotation center as tuple, (x, y)
    :return: rotated image
    """
    if center is None:
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
    else:
        image_center = center

    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result


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


def find_edge_pair(list_of_blobs, expected, axis=0, bias_left=True, distance_weight=0.5):
    distances = []

    for blob in list_of_blobs:
        avg = blob_avg(blob, axis=axis)
        support = len(blob)
        distance = abs(expected - avg)

        if bias_left and axis == 0:
            # Left sided
            if expected >= avg:
                # Blob is to the right of the expected, more focus on distance
                distances.append(support / 1.1 ** (distance * (distance_weight + 0.2)))
            else:
                # Blob is to the left of the expected, less focus on distance
                distances.append(support / 1.1 ** (distance * distance_weight))
        else:
            # Right sided
            if expected <= avg:
                # Blob is to the left of the expected, more focus on distance
                distances.append(support / 1.1 ** (distance * (distance_weight + 0.2)))
            else:
                # Blob is to the right of the expected, less focus on distance
                distances.append(support / 1.1 ** (distance * distance_weight))

    return list_of_blobs[np.argmax(distances)]


def blob_avg(blob, axis=0):
    return np.mean([edge[axis] for edge in blob])


def cluster_edges(list_of_edges, axis=0, max_clusters=14, locate_elbow=True):
    """
    Clusters a list of edges by their x-coordinate and returns a list of blobs.
    :param locate_elbow:
    :param axis: axis of clustering, 0 for x, 1 for y
    :param max_clusters:
    :param list_of_edges: List of [x, y] points to be clustered
    :returns: list of edge blobs, blob centroids
    """
    edges_x = np.array([edge[axis] for edge in list_of_edges]).reshape(-1, 1)

    kelbow = KElbowVisualizer(MiniBatchKMeans(random_state=42),
                              k=(1, min(len(list_of_edges), max_clusters)),
                              locate_elbow=locate_elbow)

    kelbow.fit(edges_x)
    labels = kelbow.estimator.fit_predict(edges_x)
    centroids = kelbow.estimator.cluster_centers_

    blobs = []
    for cluster in np.unique(labels):
        indexes = np.where(labels == cluster)[0]
        blobs.append(np.take(list_of_edges, indexes, axis=0))

    return blobs, centroids


def cluster_lines(list_of_lines, axis=0, max_clusters=14, locate_elbow=True):
    """
    Clusters a list of edges by their x-coordinate and returns a list of blobs.
    :param locate_elbow:
    :param axis: axis of clustering, 0 for x, 1 for y
    :param max_clusters:
    :param list_of_lines: List of [x1, y1, x2, y2] points to be clustered
    :returns: list of edge blobs, blob centroids
    """
    edges_x = np.array([np.mean(edge[0, [axis, axis+2]]) for edge in list_of_lines]).reshape(-1, 1)

    kelbow = KElbowVisualizer(MiniBatchKMeans(random_state=42),
                              k=(1, min(len(list_of_lines), max_clusters)),
                              locate_elbow=locate_elbow)

    kelbow.fit(edges_x)
    labels = kelbow.estimator.fit_predict(edges_x)
    centroids = kelbow.estimator.cluster_centers_

    blobs = []
    for cluster in np.unique(labels):
        indexes = np.where(labels == cluster)[0]
        blobs.append(np.take(list_of_lines, indexes, axis=0))

    return blobs, centroids