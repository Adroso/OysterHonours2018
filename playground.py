import cv2
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import skimage
import imutils
from matplotlib import pyplot as plt

image = cv2.imread('OysterImages/1 (20).JPG')
width = 19.75 #milimeters

height, width = image.shape[:2]
print(height, width)


image = imutils.rotate(image, 45)

height1, width1 = image.shape[:2]
print(height1, width1)