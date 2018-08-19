import skimage.io as ski
import skimage.filters as filter
import skimage.exposure as exposure
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import cv2
import pandas

image = ski.imread('OysterImages/1 (20).JPG', as_grey=True)
image = image[1:1000, 850:1700] #crop

image = cv2.GaussianBlur(image,(5,5), 0)
#pre processing of image
plt.imshow(image)
plt.show()
tf.reset_default_graph()