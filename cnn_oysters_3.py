import skimage.io as ski
import skimage.filters as filter
import skimage.exposure as exposure
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import cv2
import pandas

image = ski.imread('OysterImages/devided/1.jpg', as_grey=True)
#image = image[1:3000, 850:1700] #crop

#image = cv2.GaussianBlur(image,(5,5), 0)
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
result_rgb = np.zeros((187,208, 3))
result_rgb[...,0] = (result_red + (np.min(result_red)*-1) ) / (np.min(result_red)*-1 + np.max(result_red))
result_rgb[...,1] = (result_green + (np.min(result_green)*-1) ) / (np.min(result_green)*-1 + np.max(result_green))
result_rgb[...,2] = (result_blue + (np.min(result_blue)*-1) ) / (np.min(result_blue)*-1 + np.max(result_blue))
#result_rgb

horizontal_pixel_id = 0
vertical_pixel_id = 0
max_loop = 0 #425 as above
filtered_result = []
for vertical_pixel_array in result_rgb: #note the array has 3 wide values, rgb channels. these will need to be averaged then used
    #print("Current Horizontal Pixel: ", horizontal_pixel_id)
    #print(vertical_pixel_array)
    inner_list = []
    for vertical_pixel in vertical_pixel_array:
        #print(vertical_pixel)
        new_pixel_value = np.mean(vertical_pixel)
        if new_pixel_value < 0.09:
            new_pixel_value = 0
        else:
            new_pixel_value =1
        inner_list.append(new_pixel_value)

    filtered_result.append(inner_list)
    horizontal_pixel_id +=1

major_distance_count = 0
minor_distance_count = 0

oysters = []
for vp in filtered_result:
    starting_edge = -1
    ending_edge = -1

    inner_oyster = []
    for position, pixel_value in enumerate(vp[:-1]):
        if pixel_value == 1 and vp[position+1] == 0 and starting_edge == -1:
            starting_edge = position

        elif pixel_value == 1 and vp[position-1] == 0 and starting_edge > 0:
            ending_edge = position

            distance = ending_edge - starting_edge
            if distance > 1:
                major_distance_count +=1
            else:
                minor_distance_count +=1

        inner_oyster.append(pixel_value)
    oysters.append(inner_oyster)



print("majors: ", major_distance_count)
print("minors: ", minor_distance_count)

plt.imshow(oysters)
#plt.imshow(filtered_result)

plt.show()
