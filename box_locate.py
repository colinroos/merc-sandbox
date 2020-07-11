import cv2.cv2 as cv
from image_utils import *
from file_utils import *
import constants as c


def line_length(line):
    return np.sqrt((line[0, 0] - line[0, 2]) ** 2 + (line[0, 1] - line[0, 3]) ** 2)


class BoxLocate:
    def __init__(self):
        self.image = cv.imread('data/2017/images/KAD17-001_Bx1-5_11.5-25.30m_DxO.jpg')
        self.grayscale = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        self.processed = self.image.copy()
        self.processed2 = self.image.copy()
        self.out_img = self.image.copy()

        # Edges groups
        self.left_edges = []
        self.right_edges = []

        # Top Line
        self.top_edge = Line(Point(0, 0), slope=0)

        self.run()

    def run(self):
        # Process image
        self.grayscale = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)

        # self.processed = cv.adaptiveThreshold(self.grayscale, 0, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 3, 0.01)
        self.processed = cv.GaussianBlur(self.grayscale, (7, 7), 0)
        self.processed = cv.Canny(self.grayscale, 150, 190, apertureSize=3)
        self.processed = cv.GaussianBlur(self.processed, (3, 3), 0)
        # ret, self.processed2 = cv.threshold(self.grayscale, 180, 255, cv.THRESH_BINARY_INV)
        # self.processed = cv.add(self.processed, self.processed2)
        self.out_img = cv.cvtColor(self.processed, cv.COLOR_GRAY2BGR)

        v_lines = cv.HoughLinesP(self.processed, 2, np.pi, 200, 150, 100)
        h_lines = cv.HoughLinesP(self.processed, 2, np.pi/180, 200, 150, 100)

        if len(v_lines) != 0:
            for line in v_lines[:, 0, :]:
                cv.line(self.out_img, (line[0], line[1]), (line[2], line[3]), c.COLORS[1], 2)

        if len(h_lines) != 0:
            for line in h_lines[:, 0, :]:
                cv.line(self.out_img, (line[0], line[1]), (line[2], line[3]), c.COLORS[2], 2)

        # self.find_top()

        # top_edges = []
        # top_edges.extend(self.left_edges)
        # top_edges.extend(self.right_edges)

        # edges_x = [edge[0] for edge in top_edges]
        # edges_y = [edge[1] for edge in top_edges]
        # ret = np.polyfit(edges_x, edges_y, 1)

        # m, b = ret[0], ret[1]

        # self.top_edge = Line(Point(0, b), slope=m)

        # cv.line(self.out_img, (0, int(b)), (self.out_img.shape[1], int(self.out_img.shape[1] * m + b)), (0, 0, 255), 5)

        print('processed')
        cv.imshow('out', cv.resize(self.out_img, (0, 0), fx=0.25, fy=0.25))
        cv.waitKey(0)

        return self.out_img

    def find_top(self):
        left_detectors = np.arange(100, 600, 5)
        right_detectors = np.arange(self.processed.shape[1] - 600, self.processed.shape[1] - 100, 5)

        left_edges = []
        right_edges = []
        for left_detector, right_detector in zip(left_detectors, right_detectors):
            _, _, l_edges, _ = find_edge(self.processed, mode=0, detectors=left_detector, max_blob=30)
            _, _, r_edges, _ = find_edge(self.processed, mode=0, detectors=right_detector, max_blob=30)

            if len(l_edges) == 0 or len(r_edges) == 0:
                continue

            for edge in l_edges[:5]:
                left_edges.append([left_detector, edge])

            for edge in r_edges[:5]:
                right_edges.append([right_detector, edge])

        matched_blobs = []
        expected_edges = [550, 315, 315, 315]
        for list_of_edges in [left_edges, right_edges]:
            # for edge in list_of_edges:
            #     cv.circle(self.out_img, (edge[0], edge[1]), 5, (0, 255, 0), -1)

            blobs, centroids = cluster_edges(list_of_edges, axis=1, max_clusters=8, locate_elbow=True)

            # for blob, color in zip(blobs, c.COLORS[:len(blobs)]):
            #     for edge in blob:
            #         cv.circle(self.out_img, (edge[0], edge[1]), 8, color, -1)

            expected_edge = 0
            for expected, color in zip(expected_edges, c.COLORS[:len(expected_edges)]):
                expected_edge += expected
                matched_blob = find_edge_pair(blobs, axis=1, expected=expected_edge)

                expected_edge = blob_avg(matched_blob, axis=1)

                for edge in matched_blob:
                    cv.circle(self.out_img, (edge[0], edge[1]), 10, color, -1)

                matched_blobs.append(matched_blob)

        matched_blobs.sort(key=lambda x: blob_avg(x, axis=1))

        self.left_edges = matched_blobs[0]
        self.right_edges = matched_blobs[1]


if __name__ == '__main__':
    ed = BoxLocate()
    files = find_files_glob('data/2017/images/', '*.jpg')
    for file in files:
        ed.run()
        ed.image = cv.imread(file)
        pass
