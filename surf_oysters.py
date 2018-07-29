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

edges = cv2.Canny(image, 50, 50)
cv2.imshow('canny', edges)