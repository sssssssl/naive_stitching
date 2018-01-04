from sift_matcher import SiftMatcher
import cv2
import numpy as np

# naive stitching, from left to right
# left image overlays right image, this is not so good.

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
    H = sm.computeH(img_curr, img)
    offset_x, offset_y = np.dot(H, np.array([0, 0, 1])).astype(int)
    shift_x = shift_y = 0
    # transformed img start location is above or left of img_curr
    # we need to shift the img and img_curr down
    if offset_x < 0:
        H[0][1] += abs(offset_y)
        shift_x = abs(offset_x)
    if offset_y < 0:
        H[0][2] += abs(offset_y)
        shift_y = abs(offset_y)

    x_max, y_max = np.dot(H, np.array([img_width, img_height, 1])).astype(int)
    new_size = (
        max(curr_width+shift_y, x_max),
        max(curr_height+shift_y, y_max)
    )
    img_warped = cv2.warpAffine(img, H, new_size)
    img_warped[shift_y:shift_y+curr_height, shift_x:shift_x+curr_width] = img_curr
    img_curr = img_warped

cv2.imshow('res', img_curr)
cv2.waitKey(0)
cv2.destroyAllWindows()