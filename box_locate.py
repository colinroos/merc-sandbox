import cv2
from image_utils import *


class BoxLocate:
    def __init__(self):
        self.image = cv2.imread('data/2017/images/KAD17-001_Bx1-5_11.5-25.30m_DxO.jpg')
        self.out_img = self.image.copy()
        self.run()

    def run(self):
        self.out_img = self.image.copy()
        detectors = np.arange(0, self.image.shape[0], 200)

        for detector in detectors:
            _, _, edges, _ = find_edge(self.image, mode=1, detectors=detector, threshold=180)

            if len(edges) == 0:
                continue

            for edge in edges:
                cv2.circle(self.out_img, (detector, edge), 2, (0, 0, 255), -1)

        img = blur(self.image, 3)
        img = auto_canny(img)

        return img
