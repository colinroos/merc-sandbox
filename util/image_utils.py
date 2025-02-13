import numpy as np
import cv2
from sympy.geometry import Point, Line, intersection
from util import constants as c
from PyQt5 import QtGui
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Suppress future warnings on import
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

from util.elbow import KElbowVisualizer
from sklearn.cluster import KMeans, MiniBatchKMeans


def tile_images(images, mode=0):
    """
    Converts a list of images to a single image ready for display
    :param images: list-like object of images
    :param mode: horizontal or vertical tiling (default of 0 for horizontal, 1 for vertical)
    :return: concatenated image object, side-by-side
    """
    for idx, image in enumerate(images):
        if len(image.shape) == 2:
            d = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            d = image

        if idx > 0:
            if mode == 1:
                frame = cv2.vconcat([frame, d])
            else:
                frame = cv2.hconcat([frame, d])
        else:
            frame = d

    return frame


def blur(image, kernel):
    """
    Applies a Gaussian Blur to an image using the specified kernel.
    :param image: Image to be blurred, array-like object
    :param kernel: n x n kernel size
    :return: Blurred image, array-like object
    """
    img = cv2.GaussianBlur(image.copy(), (kernel, kernel), 0)
    return img


def adjustable_blur(image):
    """
    Applies an adjustable Gaussian Blur to an image. Kernel size is read from the constants.py file.
    :param image: Image to be processed, array-like object
    :return: Processed imaged, array-like object
    """
    gaussian_kernel = c.GAUSSIAN_KERNEL[c.GAUSSIAN_PARAM.value]
    img = blur(image, gaussian_kernel)
    return img


def local_hist_norm(image):
    """
    Applies a local histogram normalization using the CLAHE function as required. Specification for requirement
    is read from the constants.py file.
    :param image: Image to be processed, array-like object
    :return: Processed image, array-like object
    """
    if c.USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(image.copy())
    else:
        img = image.copy()

    return img


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

    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def transform_points(points, matrix):
    """
    Transforms a list of points by transformation matrix.
    :param points: list of points to be transformed [[x1, y1], [x2, y2]]
    :param matrix: transformation matrix, 3x3
    :return: transformed points
    """
    nd_points = np.float32(np.array(points)).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(nd_points, matrix)


def find_edge(image,
              roi=None,  # region of interest, 2 points
              mode=0,  # 0 horizontal, 1 vertical
              right_start=False,  # scan right to left if true
              threshold=100,  # binary rising/falling edge value
              detectors=None,  # list of slices to scan
              start_pixel=0,  # start scanning pixel
              end_pixel=-1,  # end scanning pixel
              max_blob=5,  # minimum length of acceptable edge
              noise_filter_length=0, ):
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

        temp_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

        for detector in detectors:
            img_slice = img[:, detector]

            # temp_img[:, detector-2:detector+2] = (0, 0, 255)
            # cv2.imwrite('./out/edge_step_0.jpg', cv2.resize(temp_img, (0, 0), fx=0.3, fy=0.3))

            # plt.plot(range(len(img_slice)), img_slice)
            # plt.xlabel('Index')
            # plt.ylabel('Brightness')
            # plt.show()
            # plt.savefig('./out/edge_step_1.jpg')

            # If start from the right, flip the slice
            if right_start:
                img_slice = img_slice[::-1]

            # Loop through sliced image and find rising and falling edges
            cropped_img_slice = img_slice[start_pixel:end_pixel - noise_filter_length]
            for idx, pixel in enumerate(cropped_img_slice):

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


def merge_to_roi(image, crop_image, roi):
    """
    Joins a cropped image with an image given the original ROI
    :param image:
    :param crop_image:
    :param roi: (x1, y1) , (x2, y2)
    :return:
    """
    img = image.copy()
    img[roi[0][0]:roi[1][0], roi[0][1]:roi[1][1]] = crop_image
    return img


def hough_intersect(image, lines):
    """
    Computes the intersection of two Hough lines.
    :param image: image to draw origin on
    :param lines: lines object, (x1, y1, x2, y2)
    :return: annotated image, origin as (x, y)
    """
    if len(image.shape) != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    try:
        if len(lines) != 0:
            # get only first and 3rd lines
            pts = []
            for x1, y1, x2, y2 in lines:
                pts.append(Line(Point(x1, y1), Point(x2, y2)))
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

            org = intersection(pts[0], pts[1])[0]

            cv2.circle(image, (org.x, org.y), 2, (0, 255, 0), -1)
        else:
            raise RuntimeError
    except RuntimeError as err:
        print('No edges or intersection found.', err)
        raise

    return image, org


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
    :param axis: axis of clustering, 0 for X, 1 for y
    :param max_clusters:
    :param list_of_edges: List of [x, y] points to be clustered
    :returns: list of edge blobs, blob centroids
    """
    # Single axis mode, deprecated
    # edges_x = np.array([edge[axis] for edge in list_of_edges]).reshape(-1, 1)

    # No edges found, return None
    if len(list_of_edges) == 0:
        return None, None

    # Combined axis mode
    scaled_edges = np.array(list_of_edges, dtype=np.float)
    scaled_edges[:, int(not axis)] = scaled_edges[:, int(not axis)] * c.EDGE_CLUSTER_SCALING

    # Declare the Elbow locator
    kelbow = KElbowVisualizer(MiniBatchKMeans(random_state=42),
                              k=(1, min(len(list_of_edges), max_clusters)),
                              locate_elbow=locate_elbow)

    # Fit the data and get the labels
    labels = kelbow.estimator.fit_predict(scaled_edges)
    centroids = kelbow.estimator.cluster_centers_

    # Build the blobs from the list of edges
    blobs = []
    for cluster in np.unique(labels):
        indexes = np.where(labels == cluster)[0]
        blobs.append(np.take(list_of_edges, indexes, axis=0))

    return blobs, centroids


def locate_edge(image, detectors, axis=0, reverse=False):
    """
    Rakes the image and locates an edge. Edges are filtered using KMeans clustering and edge blobs are returned.
    :param image: Image to be raked
    :param detectors: Indexes to rake the image at. Must be perpendicular to the axis.
    :param axis: Direction of the rake. 0 for X (top to bottom), 1 for Y (left to right)
    :param reverse: Left- or right-hand start
    :return: List of edge blobs as [X, Y] points
    """
    edges = []

    try:
        # Process all the detectors (image slices)
        for detector in detectors:
            _, _, r_blob, _ = find_edge(image,
                                        mode=axis,
                                        detectors=detector,
                                        start_pixel=0,
                                        end_pixel=600,
                                        right_start=reverse,
                                        max_blob=20,
                                        noise_filter_length=10)

            # If the length of the edges found is 0, skip this detector
            if len(r_blob) == 0:
                continue

            # Store the edges in a master list for sorting and clustering
            for edge in r_blob[:3]:
                if not axis:
                    # Horizontal edge
                    edges.append([detector, edge])
                else:
                    # Vertical edge
                    edges.append([edge, detector])

        # if axis == 1 and not reverse:
        #     temp_img = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        #     for edge in edges:
        #         cv2.circle(temp_img, (edge[0], edge[1]), 8, c.COLORS[0], -1)

            # cv2.imwrite('./out/step_2.jpg', cv2.resize(temp_img, (0, 0), fx=0.3, fy=0.3))

        # Cluster the edges in to blobs
        blobs, centroids = cluster_edges(edges, axis=int(not axis))

        # Sort the edge clusters by length (support)
        blobs.sort(key=len, reverse=True)

        if not axis:
            # Return only the first n blobs
            return blobs[:c.V_BLOBS_TO_KEEP]
        else:
            # Return only the first n blobs
            return blobs[:c.H_BLOBS_TO_KEEP]

    except TypeError as e:
        print(e)


def fit_edge(image, blob, axis=0):
    good_edge = False
    idx = len(blob)

    while not good_edge:
        # Concat all point in the blob for fitting
        temp_blob = np.concatenate(blob[:idx])

        # Extract x and y points and fit a line to them
        x = np.array([edge[int(not axis)] for edge in temp_blob], dtype=np.float)
        y = np.array([edge[axis] for edge in temp_blob], dtype=np.float)
        coefficients = np.polyfit(x, y, 1)
        m, b = coefficients[0], coefficients[1]

        # Build a Line object
        line = Line(Point(0, b), slope=m)

        if abs(m) > 0.001:
            idx -= 1
            if idx < 2:
                idx = 2
                break
        else:
            good_edge = True
            break

    if axis:
        cv2.line(image,
                 (0, int(b)),
                 (image.shape[axis], int(image.shape[axis] * m + b)),
                 c.COLORS[0],
                 10)
    else:
        cv2.line(image,
                 (int(b), 0),
                 (int(image.shape[axis] * m + b), image.shape[axis]),
                 c.COLORS[0],
                 10)

    return line, image


def four_point_transform(image, pts):
    (tl, tr, br, bl) = pts

    width_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = np.max(int(width_A), int(width_B))

    height_A = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_B = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = np.max(int(height_A), int(height_B))

    dst = np.array([[0, 0],
                    [max_width - 1, 0],
                    [max_width - 1, max_height - 1],
                    [0, max_height - 1]],
                   dtype=np.float)

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped
