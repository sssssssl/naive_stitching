from sift_matcher import SiftMatcher
import cv2
import numpy as np

# naive stitching, from left to right
# constantly transform left image to right image space
# and attach right image to current warped-image

sm = SiftMatcher()

path_list = [
    './images/yosemite1.jpg',
    './images/yosemite2.jpg',
    './images/yosemite3.jpg',
    './images/yosemite4.jpg',
]

img_list = []

for p in path_list:
    img = cv2.imread(p)
    img_list.append(img)

img_curr = img_list[0]
for img in img_list[1:]:
    img_height, img_width = img.shape[:2]
    curr_height, curr_width = img_curr.shape[:2]
    # h transforms img_curr to img
    H = sm.computeH(img, img_curr)
    offset_x, offset_y = np.dot(H, np.array([0, 0, 1])).astype(int)
    shift_x = shift_y = 0
    # transformed img_curr start location is above or left of img
    # we need to shift the img and img_curr right or down
    # add translation component to affine matrix to shift images.
    if offset_x < 0:
        H[0][2] += abs(offset_x)
        shift_x = abs(offset_x)
    if offset_y < 0:
        H[1][2] += abs(offset_y)
        shift_y = abs(offset_y)

    x_max, y_max = np.dot(H, np.array([curr_width, curr_height, 1])).astype(int)
    new_size = (
        max(img_width+shift_x, x_max),
        max(img_height+shift_y, y_max)
    )
    img_warped = cv2.warpAffine(img_curr, H, new_size)
    # here we do direct stitching, but actually need blending.
    img_warped[shift_y:shift_y+img_height, shift_x:shift_x+img_width] = img
    img_curr = img_warped

cv2.imshow('res', img_curr)
cv2.waitKey(0)
cv2.destroyAllWindows()