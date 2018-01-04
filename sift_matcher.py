import cv2
import numpy as np


class SiftMatcher(object):
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.bf = cv2.BFMatcher_create()

    def detect_sift(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, des = self.sift.detectAndCompute(gray, None)
        return (kps, des)

    def bfMatch(self, sift1, sift2):
        """
        Given SIFT attributes of 2 images(keypoint list and descriptor list), 
        match keypoints according to descriptor distance,
        sort matches from the closet match to the farest match,
        return 2 arrays, each containing positions of the matched keypoints.
        """
        kps1, des1 = sift1
        kps2, des2 = sift2
        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        # list containing (x, y) of keypoints
        kp1_pos = []
        kp2_pos = []
        for m in matches[:30]:
            kp1, kp2 = kps1[m.queryIdx], kps2[m.trainIdx]
            kp1_pos.append(kp1.pt)
            kp2_pos.append(kp2.pt)
        # convert python list to numpy int array
        kp1_pos = np.array(kp1_pos).astype(int)
        kp2_pos = np.array(kp2_pos).astype(int)
        return (kp1_pos, kp2_pos)

    def fitAffineMatrix(self, kp1_pos, kp2_pos):
        """
        least square fit 2 equations:
            1. `x = H11 * x' + H12 * y' + H13`
            2. `y = H21 * x' + H22 * y' + H23`
        """
        A = np.c_[kp2_pos, np.ones(len(kp2_pos))]  # [x', y', 1]
        b_1 = kp1_pos[:, 0]  # x
        h11, h12, h13 = np.linalg.lstsq(A, b_1)[0]
        b_2 = kp1_pos[:, 1]  # y
        h21, h22, h23 = np.linalg.lstsq(A, b_2)[0]
        H = np.array([
            [h11, h12, h13],
            [h21, h22, h23]
        ])
        return H

    def computeH(self, img1, img2):
        """
        compute transformation matrix from img2 to img1
        """
        sift1 = self.detect_sift(img1)
        sift2 = self.detect_sift(img2)
        kp1_pos, kp2_pos = self.bfMatch(sift1, sift2)
        H = self.fitAffineMatrix(kp1_pos, kp2_pos)
        return H


if __name__ == '__main__':
    pass
