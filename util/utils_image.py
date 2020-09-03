import numpy as np
import cv2
from sympy.geometry import Point, Line, intersection
from util import constants as c
from PyQt5 import QtGui
from scipy.signal import find_peaks

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


def pre_process_center(image):
    """
    Applies a customized pre-processing filter to a specified image for center finding.
    :param image: Image to be processed, array-like object
    :return: Process image, array-like object
    """
    if c.USE_CLAHE_CENTER:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(image.copy())
    else:
        img = image.copy()

    img = blur(img, 11)

    img = cv2.Canny(img, threshold1=0, threshold2=30, apertureSize=3)

    return img


def pre_process_slit(image, slit_threshold):
    # Convert the image to a binary image using a threshold
    _, img = cv2.threshold(image, slit_threshold, 255, 0)

    # Invert the image
    img = cv2.bitwise_not(img)

    # Perform a dilation to expand the opening 1 pixel
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, (3, 3))

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


def draw_crosshairs(image,
                    transform=None,
                    width=100,
                    height=100,
                    x_offset=0,
                    y_offset=0):
    """
    Draw crosshairs on image at a given location
    :param image:
    :param transform:
    :param width:
    :param height:
    :param x_offset:
    :param y_offset:
    :return:
    """
    if transform is None:
        transform = np.identity(3, dtype=np.float32)

    p1 = np.array([height / 2 + x_offset, 0])
    p2 = np.array([height / 2 + x_offset, width])
    p3 = np.array([0, width / 2 + y_offset])
    p4 = np.array([height, width / 2 + y_offset])
    pts = np.float32([p1, p2, p3, p4]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, transform)

    line1 = Line(Point(dst[0, 0]), Point(dst[1, 0]))
    line2 = Line(Point(dst[2, 0]), Point(dst[3, 0]))
    origin = intersection(line1, line2)[0]

    image = cv2.line(image, (line1.p1.x, line1.p1.y), (line1.p2.x, line1.p2.y), (0, 255, 0), 1)
    image = cv2.line(image, (line2.p1.x, line2.p1.y), (line2.p2.x, line2.p2.y), (0, 255, 0), 1)

    return image, origin


def match_template(template,
                   image,
                   draw_crosshairs_flag=True,
                   draw_matches=False):
    """
    Matches a template to an image, returning the transformation matrix and the plotted image with crosshairs
    :param template: template image to be matched
    :param image: image to determine transformation of
    :param draw_crosshairs_flag: Flag to draw crosshairs
    :param draw_matches: Flag to display matches
    :return: Image with matched crosshairs, 3x3 Transformation Matrix
    """
    # Declare OBR feature detector
    orb = cv2.ORB_create(nfeatures=500)
    x_offset = 5
    y_offset = 10
    w, h = template.shape[:2]

    # Compute key points and descriptions
    kp_template, desc_template = orb.detectAndCompute(template, None)
    kp_frame, desc_frame = orb.detectAndCompute(image, None)

    try:
        if len(kp_frame) > 10:
            template = draw_crosshairs(template, width=w, height=h, x_offset=5, y_offset=10)

            # Find brute force matches
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(desc_template, desc_frame)

            if len(matches) != 0:
                # Extract points
                sorted_matches = sorted(matches, key=lambda x: x.distance)
                src_pts = np.float32([kp_template[m.queryIdx].pt for m in sorted_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in sorted_matches]).reshape(-1, 1, 2)

                # Find transformation matrix
                transform, mask = cv2.estimateAffine2D(src_pts, dst_pts)
                transform = np.vstack([transform, [0, 0, 1]])

                # Convert image to color
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                if draw_crosshairs_flag:
                    image, origin = draw_crosshairs(image, transform, w, h, x_offset=x_offset, y_offset=y_offset)
                else:
                    _, origin = draw_crosshairs(image, transform, w, h, x_offset=x_offset, y_offset=y_offset)

                if draw_matches:
                    image = cv2.drawMatches(template, kp_template, image, kp_frame, sorted_matches[:20], None, flags=2)

                return image, transform, origin, True
        else:
            return image, None, None, False
    except RuntimeError as err:
        print('No matches found', err)
        raise


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


def draw_color_tile(widget, color):
    width = widget.frameGeometry().width()
    height = widget.frameGeometry().height()
    bytes_per_line = 3 * width
    img = np.zeros((height, width, 3), np.uint8)
    img[:] = color
    qimg = QtGui.QImage(img.data, width, height, bytes_per_line,
                        QtGui.QImage.Format_RGB888).rgbSwapped()
    return QtGui.QPixmap.fromImage(qimg)


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