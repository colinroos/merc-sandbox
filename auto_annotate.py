import cv2.cv2 as cv
import numpy as np
import util.constants as c
from util.files import *
from util.image_utils import *
from sympy.geometry import Line, Point


class AutoAnnotate:
    def __init__(self, images):
        # Image objects
        self.image_index = 0
        self.prev_index = 0
        self.images = images
        self.image = cv.imread(images[self.image_index])
        self.image = cv.resize(self.image, c.NOMINAL_IMG_DIMS)
        self.processed_image = self.image.copy()
        self.out_image = self.image.copy()

        # Parameters
        self.block_size = 2
        self.kernel_size = 5
        self.free_parameter = 0.07
        self.threshold = 0.01

        # Edge Blobs
        self.top_blob = []
        self.left_blob = []
        self.right_blob = []
        self.bottom_blob = []

        # Edge Objects
        self.left_edge = Line(Point(0, 0), slope=0)
        self.right_edge = Line(Point(0, 0), slope=0)
        self.top_edge = Line(Point(0, 0), slope=0)
        self.bottom_edge = Line(Point(0, 0), slope=0)

        # Corner Objects
        self.top_left_corner = Point(0, 0)
        self.top_right_corner = Point(0, 0)
        self.bottom_left_corner = Point(0, 0)
        self.bottom_right_corner = Point(0, 0)

    def find_corners(self):
        """
        Locates all Harris corners in the image.
        :return:
        """
        result = cv.cornerHarris(self.processed_image, self.block_size, self.kernel_size, self.free_parameter)

        # Results are marked through the dialated corners
        result = cv.dilate(result, None)

        # Draw the corners using a threshold
        self.out_image[result > self.threshold * result.max()] = [0, 0, 255]

    def fit_edges(self):
        # Top Edge
        self.top_edge, self.out_image = fit_edge(self.out_image, self.top_blob, axis=1)

        # Left Edge
        self.left_edge, self.out_image = fit_edge(self.out_image, self.left_blob, axis=0)

        cv2.imwrite('./out/step_4.jpg', cv2.resize(self.out_image, (0, 0), fx=0.3, fy=0.3))

        # Right Edge
        self.right_edge, self.out_image = fit_edge(self.out_image, self.right_blob, axis=0)

        # Bottom Edge
        self.bottom_edge, self.out_image = fit_edge(self.out_image, self.bottom_blob, axis=1)

        # Fit Corner Intersections
        self.top_left_corner = self.left_edge.intersection(self.top_edge)[0]
        self.top_right_corner = self.right_edge.intersection(self.top_edge)[0]
        self.bottom_left_corner = self.left_edge.intersection(self.bottom_edge)[0]
        self.bottom_right_corner = self.right_edge.intersection(self.bottom_edge)[0]

        # cv2.imwrite('./out/step_5.jpg', cv2.resize(self.out_image, (0, 0), fx=0.3, fy=0.3))

    def find_top(self):
        detectors = np.concatenate([np.arange(0, 300, 10),
                                    np.arange(3000, 3300, 10)])

        blobs = locate_edge(self.processed_image,
                            detectors=detectors,
                            axis=0,
                            reverse=False)

        for blob, color in zip(blobs, c.COLORS[:len(blobs)]):
            for edge in blob:
                cv.circle(self.out_image, (edge[0], edge[1]), 8, color, -1)

        # Assign blob cluster to global object
        self.top_blob = blobs.copy()

    def find_left(self):
        detectors = np.arange(0, 2300, 10)

        blobs = locate_edge(self.processed_image,
                            detectors=detectors,
                            axis=1,
                            reverse=False)

        for blob, color in zip(blobs, c.COLORS[:len(blobs)]):
            for edge in blob:
                cv.circle(self.out_image, (edge[0], edge[1]), 8, color, -1)

        # cv2.imwrite('./out/step_3.jpg', cv2.resize(self.out_image, (0, 0), fx=0.3, fy=0.3))

        # Assign blob cluster to global object
        self.left_blob = blobs.copy()

    def find_right(self):
        detectors = np.arange(0, 2300, 10)

        blobs = locate_edge(self.processed_image,
                            detectors=detectors,
                            axis=1,
                            reverse=True)

        for blob, color in zip(blobs, c.COLORS[:len(blobs)]):
            for edge in blob:
                cv.circle(self.out_image, (edge[0], edge[1]), 8, color, -1)

        # Assign blob cluster to global object
        self.right_blob = blobs.copy()

    def find_bottom(self):
        detectors = np.concatenate([np.arange(0, 300, 10),
                                    np.arange(3000, 3300, 10)])

        blobs = locate_edge(self.processed_image,
                            axis=0,
                            detectors=detectors,
                            reverse=True)

        # Annotate the blobs on the image
        for blob, color in zip(blobs, c.COLORS[:len(blobs)]):
            for edge in blob:
                cv.circle(self.out_image, (edge[0], edge[1]), 8, color, -1)

        # Store the blobs in a global object for use later
        self.bottom_blob = blobs.copy()

    def pre_process(self):
        """
        Pre-processes the raw image for further processing and analysis.
        :return:
        """
        self.processed_image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        # cv.imwrite('./out/step_0.jpg', cv.resize(self.image, (0, 0), fx=0.3, fy=0.3))
        # cv.imwrite('./out/step_1.jpg', cv.resize(self.processed_image, (0, 0), fx=0.3, fy=0.3))
        self.out_image = self.image.copy()
        # self.processed_image = cv.GaussianBlur(self.processed_image, 3, 0.33)

    def load_image(self):
        if self.image_index != self.prev_index:
            self.image = cv.imread(self.images[self.image_index])
            self.image = cv.resize(self.image, c.NOMINAL_IMG_DIMS)

    def run(self):
        """
        Runs the inspection pipeline.
        :return:
        """
        self.load_image()
        self.pre_process()
        # self.find_corners()
        self.find_top()
        self.find_left()
        self.find_right()
        self.find_bottom()
        # self.fit_edges()


if __name__ == '__main__':
    files = find_files_glob('data/2017/images/', '*.jpg')
    ed = AutoAnnotate(files)
    for file in files:
        ed.run()
        ed.image = cv.imread(file)
        break
