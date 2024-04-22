# 将一个图像的三个通道分别显示出来

import cv2
import numpy as np


def show_channel(image):
    b, g, r = cv2.split(image)
    cv2.imshow("Blue", b)
    cv2.imshow("Green", g)
    cv2.imshow("Red", r)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image = cv2.imread("stenosis_data/train/images/1.png")
show_channel(image)
