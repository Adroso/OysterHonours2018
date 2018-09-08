import cv2
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import skimage
import imutils
from matplotlib import pyplot as plt

#load image and specify width of the left most object
image = cv2.imread('OysterImages/1 (21).JPG')
image = image[1:1000, 850:1700]

edged = cv2.Canny(image, 250, 250)
test = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)

cv2.imshow('edges', test)
cv2.waitKey(0)

cv2.imshow("thing", image)
cv2.waitKey(0)