import cv2.cv2 as cv
from util.image_processing import *
from util.files import *
import constants as c
from sympy.geometry import Point, Line


def line_length(line):
    return np.sqrt((line[0, 0] - line[0, 2]) ** 2 + (line[0, 1] - line[0, 3]) ** 2)


def line_blob_avg(line_blob, axis=0):

    # coords = []
    # for x in line_blob:
    #     coords.extend(x[axis])
    #     coords.extend(x[axis + 2])
    coords = [x[0, [axis, axis + 2]] for x in line_blob]
    avg_line = np.mean(coords)

    return avg_line


def filter_lines_by_slope(lines, axis=0):
    if axis == 0:
        max_slope = 0.3
    else:
        max_slope = 10000

    filtered = []
    for line in lines:
        p1 = Point(line[0, 0], line[0, 1])
        p2 = Point(line[0, 2], line[0, 3])

        l = Line(p1, p2)
        if abs(l.slope) <= max_slope and axis == 0:
            filtered.append(line)
        elif abs(l.slope) >= max_slope and axis == 1:
            filtered.append(line)

    return np.array(filtered)


def line_avg_slope(list_of_lines):
    slopes = []
    for line in list_of_lines:
        p1 = Point(line[0, 0], line[0, 1])
        p2 = Point(line[0, 2], line[0, 3])

        l = Line(p1, p2)
        slopes.append(l.slope.evalf())

    return np.mean(slopes, dtype=np.float)


def rank_line_blob(list_of_blobs, axis=0, reverse=False):
    list_of_blobs_sorted = list_of_blobs.copy()
    list_of_blobs_sorted.sort(key=lambda x: line_blob_avg(x, axis), reverse=reverse)

    blob_scores = []
    for idx, blob in enumerate(list_of_blobs_sorted):
        score = (len(list_of_blobs_sorted) / (idx + 1)) * 1/np.mean(np.var(blob, axis=0))
        blob_scores.append(score)

    return list_of_blobs_sorted[np.argmax(blob_scores)]


class BoxLocate:
    def __init__(self):
        self.image = cv.imread('data/2017/images/KAD17-001_Bx1-5_11.5-25.30m_DxO.jpg')
        self.grayscale = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        self.processed = self.image.copy()
        self.out_img = self.image.copy()

        # Edges groups
        self.left_edges = []
        self.right_edges = []

        # Top Line
        self.top_edge = Line(Point(0, 0), slope=0)

        self.run()

    def run(self):
        # Resize the image to a nominal size
        self.image = cv.resize(self.image, c.NOMINAL_IMG_DIMS)
        # cv.imshow('', cv.resize(self.image, (0, 0), fx=0.25, fy=0.25))
        # cv.waitKey(0)

        # Apply a local histogram normalization
        self.processed = cv.cvtColor(self.image, cv.COLOR_BGR2YCrCb)
        color_planes = cv.split(self.processed)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        color_planes[0] = clahe.apply(color_planes[0])
        self.processed = cv.merge(color_planes)
        self.processed = cv.cvtColor(self.processed, cv.COLOR_YCrCb2BGR)
        # cv.imshow('', cv.resize(self.processed, (0, 0), fx=0.25, fy=0.25))
        # cv.waitKey(0)

        # Blur the image to extrapolate some noise
        self.processed = cv.GaussianBlur(self.processed, (11, 11), 0)
        # cv.imshow('', cv.resize(self.processed, (0, 0), fx=0.25, fy=0.25))
        # cv.waitKey(0)

        # Edge detect using Canny
        self.processed = cv.Canny(self.processed, 60, 150, apertureSize=3)
        # cv.imshow('', cv.resize(self.processed, (0, 0), fx=0.25, fy=0.25))
        # cv.waitKey(0)

        # Blur the detected edges to extrapolate some antialiasing noise
        self.processed = cv.GaussianBlur(self.processed, (3, 3), 0)
        # cv.imshow('', cv.resize(self.processed, (0, 0), fx=0.25, fy=0.25))
        # cv.waitKey(0)

        # Create an output image for annotations
        self.out_img = self.image.copy()

        # Extract horizontal and vertical lines
        v_lines = cv.HoughLinesP(self.processed, 1, np.pi, threshold=300, minLineLength=200, maxLineGap=10)
        h_lines = cv.HoughLinesP(self.processed, 1, np.pi/180, threshold=400, minLineLength=400, maxLineGap=10)

        if v_lines is not None and h_lines is not None:

            # Filter the lines by the expected slope value
            v_lines = filter_lines_by_slope(v_lines, axis=1)
            h_lines = filter_lines_by_slope(h_lines, axis=0)

            # Cluster the lines into edges
            v_edges, v_centroids = cluster_lines(v_lines, axis=0, max_clusters=5, locate_elbow=False)
            h_edges, h_centroids = cluster_lines(h_lines, axis=1, max_clusters=30, locate_elbow=False)

            # Draw the edges on the output image

            # Loop through each blob to draw it a different color

            # Get the average slope of the horizontal lines to correct image rotation
            avg_slope = line_avg_slope(h_lines)

            # Convert the slope to a rotation in degrees
            theta = np.degrees(np.arctan(avg_slope))

            # Rank the blob for the best horizontal and vertical edges
            matched_v_blob_1 = rank_line_blob(v_edges, axis=0)
            matched_v_blob_2 = rank_line_blob(v_edges, axis=0, reverse=True)
            matched_h_blob_1 = rank_line_blob(h_edges, axis=1)
            matched_h_blob_2 = rank_line_blob(h_edges, axis=1, reverse=True)

            for matched_blob in [matched_v_blob_1, matched_h_blob_1, matched_v_blob_2, matched_h_blob_2]:
                for line in matched_blob[:, 0, :]:
                    cv.line(self.out_img, (line[0], line[1]), (line[2], line[3]), c.COLORS[2], 10)

            # Rotate the image to straighten the boxes
            self.out_img = rotate_image(self.out_img, theta)

        print('processed')
        cv.imshow('out', cv.resize(self.out_img, (0, 0), fx=0.25, fy=0.25))
        cv.waitKey(0)

        return self.out_img


if __name__ == '__main__':
    ed = BoxLocate()
    files = find_files_glob('data/2017/images/', '*.jpg')
    for file in files:
        ed.run()
        ed.image = cv.imread(file)
        break
