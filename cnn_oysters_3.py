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

#pre processing of image
plt.imshow(image)
plt.show()
tf.reset_default_graph()

# Write the kernel weights as a 2D array.
kernel_h = np.array([3, 3])
kernel_h = [ [-1,-2,-1], [0,0,0], [1,2,1] ]
kernel_v = np.array([3, 3])
kernel_v = [ [-1,0,1], [-2,0,2], [-1,0,1] ]


# Kernel weights
if len(kernel_h) == 0 or len(kernel_v) == 0:
    print('Please specify the kernel!')

input_placeholder = tf.placeholder(
    dtype=tf.float32, shape=(1, image.shape[0], image.shape[1], 1))
with tf.name_scope('convolution'):
    conv_w_h = tf.constant(kernel_h, dtype=tf.float32, shape=(3, 3, 1, 1))
    conv_w_v = tf.constant(kernel_v, dtype=tf.float32, shape=(3, 3, 1, 1))
    output_h = tf.nn.conv2d(input=input_placeholder, filter=conv_w_h, strides=[1, 1, 1, 1], padding='SAME')
    output_v = tf.nn.conv2d(input=input_placeholder, filter=conv_w_v, strides=[1, 1, 1, 1], padding='SAME')
    output_h = tf.layers.max_pooling2d(output_h, 2, 2)
    output_v = tf.layers.max_pooling2d(output_v, 2, 2)


with tf.Session() as sess:
    result_h = sess.run(output_h, feed_dict={
            input_placeholder: image[np.newaxis, :, :, np.newaxis]})
    result_v = sess.run(output_v, feed_dict={
            input_placeholder: image[np.newaxis, :, :, np.newaxis]})

result_lenght = ((result_v**2) + (result_h**2))**0.5
plt.imshow(result_lenght[0, :, :, 0], cmap='hot')

result_angle = (np.arctan(result_v/(result_h+0.00000001)))#*(2*math.pi)
plt.imshow(result_angle[0, :, :, 0], cmap='hot')

#normalize like crazy
result_lenght_norm = (result_lenght[0,:,:,0] + (np.min(result_lenght)*-1) ) / (np.min(result_lenght)*-1 + np.max(result_lenght))
result_angle_norm = result_angle[0,:,:,0]
result_red = np.absolute(result_lenght_norm * np.cos(result_angle_norm+4.2))
result_green = np.absolute(result_lenght_norm * np.cos(result_angle_norm+2.1))
result_blue = np.absolute(result_lenght_norm * np.cos(result_angle_norm))
result_rgb = np.zeros((499,425, 3))
result_rgb[...,0] = (result_red + (np.min(result_red)*-1) ) / (np.min(result_red)*-1 + np.max(result_red))
result_rgb[...,1] = (result_green + (np.min(result_green)*-1) ) / (np.min(result_green)*-1 + np.max(result_green))
result_rgb[...,2] = (result_blue + (np.min(result_blue)*-1) ) / (np.min(result_blue)*-1 + np.max(result_blue))
#result_rgb

i = 1
for vertical_pixel_array in result_rgb: #note the array has 3 wide values, rgb channels. these will need to be averaged then used
    d = pandas.DataFrame(vertical_pixel_array)
    d.to_csv("F:\\Honours Image Library\\CSVs\\new{}.csv".format(i))
    i+= 1


#plt.imshow(result_rgb)

plt.show()
