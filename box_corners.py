import cv2.cv2 as cv
import numpy as np
import util.constants as c
from util.files import *
from util.utils_image import *
from sympy.geometry import Line, Point


class BoxCorners:
    def __init__(self, image):
        self.image = cv.imread(image)
        self.processed_image = self.image.copy()
        self.out_image = self.image.copy()

        # Parameters
        self.block_size = 2
        self.kernel_size = 5
        self.free_parameter = 0.07
        self.threshold = 0.01

        self.left_edge = Line(Point(0, 0), slope=0)
        self.right_edge = Line(Point(0, 0), slope=0)
        self.top_edge = Line(Point(0, 0), slope=0)
        self.bottom_edge = Line(Point(0, 0), slope=0)

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

    def find_top(self):
        # detectors = np.arange(100, 400, 10)
        detectors = np.concatenate([np.arange(100, 400, 10),
                                    np.arange(4600, 5100, 10)])

        edges = []

        for detector in detectors:
            _, _, r_blob, _ = find_edge(self.processed_image,
                                        mode=0,
                                        detectors=detector,
                                        start_pixel=0,
                                        end_pixel=600,
                                        right_start=False,
                                        max_blob=20,
                                        noise_filter_length=10)

            # If the length of the edges found is 0, skip this detector
            if len(r_blob) == 0:
                continue

            for edge in r_blob[:3]:
                edges.append([detector, edge])

        # cluster the edges in to blobs
        blobs, centroids = cluster_edges(edges, axis=1)

        # sort the edge clusters by length (support)
        blobs.sort(key=len, reverse=True)

        for blob, color in zip(blobs[:5], c.COLORS[:len(blobs[:5])]):
            for edge in blob:
                cv.circle(self.out_image, (edge[0], edge[1]), 8, color, -1)

    def find_left(self):
        detectors = np.arange(0, 3600, 10)

        edges = []

        for detector in detectors:
            _, _, r_blob, _ = find_edge(self.processed_image,
                                        mode=1,
                                        detectors=detector,
                                        start_pixel=0,
                                        end_pixel=200,
                                        right_start=False,
                                        max_blob=20,
                                        noise_filter_length=10)

            if len(r_blob) == 0:
                continue

            for edge in r_blob[:3]:
                edges.append([edge, detector])

        blobs, centroids = cluster_edges(edges, axis=0)

        blobs.sort(key=len, reverse=True)

        for blob, color in zip(blobs[:5], c.COLORS[:len(blobs[:5])]):
            for edge in blob:
                cv.circle(self.out_image, (edge[0], edge[1]), 8, color, -1)

    def find_right(self):
        detectors = np.arange(0, 3600, 10)

        edges = []

        for detector in detectors:
            _, _, r_blob, _ = find_edge(self.processed_image,
                                        mode=1,
                                        detectors=detector,
                                        start_pixel=0,
                                        end_pixel=200,
                                        right_start=True,
                                        max_blob=20,
                                        noise_filter_length=10)

            if len(r_blob) == 0:
                continue

            for edge in r_blob[:3]:
                edges.append([edge, detector])

        blobs, centroids = cluster_edges(edges, axis=0)

        blobs.sort(key=len, reverse=True)

        for blob, color in zip(blobs[:5], c.COLORS[:len(blobs[:5])]):
            for edge in blob:
                cv.circle(self.out_image, (edge[0], edge[1]), 8, color, -1)

    def pre_process(self):
        """
        Pre-processes the raw image for further processing and analysis.
        :return:
        """
        self.processed_image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        # self.processed_image = cv.GaussianBlur(self.processed_image, 3, 0.33)

    def run(self):
        """
        Runs the inspection pipeline.
        :return:
        """
        self.pre_process()
        # self.find_corners()
        self.find_top()
        self.find_left()
        self.find_right()


if __name__ == '__main__':
    files = find_files_glob('data/2017/images/', '*.jpg')
    ed = BoxCorners(files[0])
    for file in files:
        ed.run()
        ed.image = cv.imread(file)
        break
