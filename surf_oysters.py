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

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(1,1))
gray = clahe.apply(gray)

edges = cv2.SURF
cv2.imshow('canny', edges)