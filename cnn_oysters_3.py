import skimage.io as ski
import skimage.filters as filter
import skimage.exposure as exposure
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import cv2
import pandas
import lensfunpy

#Setting all globals and constants

#THE MAIN ONE
PIXEL_VALUE_TO_ACTUAL_VALUE_FACTOR = 0.11

#For pixel counting Algorithim
PIXEL_TO_LOOK = 1
INVERSE_PIXEL_TO_LOOK = 0
DISTANCE_THRESHOLD = 40
PIXEL_IGNORE_THRESHOLD = 30

#For Lens Correction
cam_maker = 'GoPro'
cam_model = 'HERO4 Silver'
lens_maker = 'GoPro'
lens_model = 'HERO4'
focal_length = 5
apperture = 2.97

#For Cropping
left_crop = 900
right_crop = 1600
top_crop = 50
bottom_crop = 2900
NUMBER_OF_OYSTERS_HIGH = 8
NUMBER_OF_OYSTERS_WIDE = 2
# Note built for a max of 2 wide, if this needs to be changed for more than 2 code in the ROI section deeds to be edited

"""Actual Code Now"""


"""PRE-PROCESSING SECTION"""
#Reading Image
raw_image = cv2.imread('OysterImages/1 (20).JPG')
grey_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
height, width = grey_image.shape[0], grey_image.shape[1]


#Lens Correction
db = lensfunpy.Database()
cam = db.find_cameras(cam_maker, cam_model)[0]
lens = db.find_lenses(cam, lens_maker, lens_model)[0]
mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)
mod.initialize(focal_length, apperture, 1)
undist_coords = mod.apply_geometry_distortion()
grey_image_undistorted = cv2.remap(grey_image, undist_coords, None, cv2.INTER_LANCZOS4)

#Blur
grey_image_undistorted = cv2.GaussianBlur(grey_image_undistorted,(5,5), 1)

#Rotation Correction
if width > height: #if image is landscape (meaning oyster hinges are sideways)
    rows, cols = grey_image_undistorted.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
    grey_rotated_undistort_image = cv2.warpAffine(grey_image_undistorted, rotation_matrix, (cols, rows))
else:
    grey_rotated_undistort_image = grey_image_undistorted

#Region of Interest 1
full_pre_processed_image = grey_rotated_undistort_image[top_crop:bottom_crop, left_crop:right_crop] #crop
#[top:bottom, left:right]

#Region of Interest 2
separated_oyster_images = {}
cropped_h, cropped_w = full_pre_processed_image.shape[0], full_pre_processed_image.shape[1]
roi_grid_h_factor = int(cropped_h/NUMBER_OF_OYSTERS_HIGH)
roi_grid_w_factor = int(cropped_w/NUMBER_OF_OYSTERS_WIDE)

roi_counter = 0 #starts at 1 to allow multipication of it, make sure the condition is a < number+1
while roi_counter < NUMBER_OF_OYSTERS_HIGH:
    separated_oyster_images[str(roi_counter) + 'A'] = full_pre_processed_image[(roi_grid_h_factor*roi_counter):(roi_grid_h_factor*(roi_counter+1)), 0:roi_grid_w_factor]
    separated_oyster_images[str(roi_counter) + 'B'] = full_pre_processed_image[(roi_grid_h_factor*roi_counter):(roi_grid_h_factor*(roi_counter+1)), roi_grid_w_factor:]
    roi_counter+=1


#show an image
plt.imshow(separated_oyster_images['1A'])
plt.show()
tf.reset_default_graph()

"""END OF PRE-PROCESSING SECTION"""

"""FOR THE PURPOSE OF RUNNING WHILE TESTING
SPECIFYING 1 OYSTER HERE"""

image = separated_oyster_images['1A']

"""CNN EDGE DETECTION SECTION"""
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


result_lenght_norm = (result_lenght[0,:,:,0] + (np.min(result_lenght)*-1) ) / (np.min(result_lenght)*-1 + np.max(result_lenght))
result_angle_norm = result_angle[0,:,:,0]
result_red = np.absolute(result_lenght_norm * np.cos(result_angle_norm+4.2))
result_green = np.absolute(result_lenght_norm * np.cos(result_angle_norm+2.1))
result_blue = np.absolute(result_lenght_norm * np.cos(result_angle_norm))
result_rgb = np.zeros((len(result_red),len(result_red[0]), 3))
result_rgb[...,0] = (result_red + (np.min(result_red)*-1) ) / (np.min(result_red)*-1 + np.max(result_red))
result_rgb[...,1] = (result_green + (np.min(result_green)*-1) ) / (np.min(result_green)*-1 + np.max(result_green))
result_rgb[...,2] = (result_blue + (np.min(result_blue)*-1) ) / (np.min(result_blue)*-1 + np.max(result_blue))


"""END OF CNN-EDGE DECTECTION SECTION"""

"""FILTERING WEAK EDGES SECTION"""
#filtering the list for stronger values
horizontal_pixel_id = 0
vertical_pixel_id = 0
max_loop = 0 #425 as above
filtered_result = []
for vertical_pixel_array in result_rgb: #note the array has 3 wide values, rgb channels. these will need to be averaged then used
    #print("Current Horizontal Pixel: ", horizontal_pixel_id)
    #print(vertical_pixel_array)
    inner_list = []
    for vertical_pixel in vertical_pixel_array:
        new_pixel_value = np.mean(vertical_pixel) #might change this way of converting rgb into a single channel
        if new_pixel_value < 0.079:
            new_pixel_value = 0
        else:
            new_pixel_value =1
        inner_list.append(new_pixel_value)

    filtered_result.append(inner_list)
    horizontal_pixel_id +=1
filtered_results_2 = np.array(filtered_result).transpose().tolist()
#plt.imshow(filtered_result)
"""END OF FILTERING EDGES SECTION"""


"""PIXEL COUNTING ALGORITHIM SECTION"""
major_distance_count = 0
minor_distance_count = 0

# coutning horizontal distances
hp_max = [0] #[distance, outerlistposition, start_innerlist, end_innerlist]
test_max = [0,0,0,0]
loop_count = 0
for po, hp in enumerate(filtered_result):
    starting_edge = -1
    ending_edge = -1

    inner_oyster = []

    for position, pixel_value in enumerate(hp[:-1]):
        if pixel_value == PIXEL_TO_LOOK and hp[position + 1] == INVERSE_PIXEL_TO_LOOK and starting_edge == -1:
            starting_edge = position
        elif pixel_value == PIXEL_TO_LOOK and hp[position - 1]== INVERSE_PIXEL_TO_LOOK and starting_edge != -1:
            ending_edge = position
            distance = ending_edge - starting_edge
            if distance > DISTANCE_THRESHOLD:
                hp[starting_edge:ending_edge + 1] = [1] * ((ending_edge + 1) - starting_edge)
                major_distance_count +=1
                if distance > hp_max[0] and loop_count < len(result_red) - PIXEL_IGNORE_THRESHOLD:
                    hp_max[0] = distance
                    hp[starting_edge:ending_edge + 1] = [0.7] * ((ending_edge + 1) - starting_edge)
                    #print("Current Max Distance APM: ", distance)
                    test_max = [po,starting_edge, ending_edge, distance]
            else:
                hp[starting_edge:ending_edge + 1] = [0] * ((ending_edge + 1) - starting_edge)
                minor_distance_count +=1

            starting_edge = -1
            ending_edge = -1
        else:
            hp[position] = 0

    loop_count +=1

#rint(test_max)
#plt.imshow(filtered_result)

#counting vertical distances
vp_max = [0] #[distance, outerlistposition, start_innerlist, end_innerlist]
test_max_2 = [0,0,0,0]
loop_count_vp = 0
for vp_po, vp in enumerate(filtered_results_2):
    starting_edge = -1
    ending_edge = -1
    inner_oyster = []

    for position, pixel_value in enumerate(vp[:-1]):
        if pixel_value == PIXEL_TO_LOOK and vp[position + 1] == INVERSE_PIXEL_TO_LOOK and starting_edge == -1:
            starting_edge = position
        elif pixel_value == PIXEL_TO_LOOK and vp[position - 1]==INVERSE_PIXEL_TO_LOOK and starting_edge != -1:
            ending_edge = position
            distance = ending_edge - starting_edge
            if distance > DISTANCE_THRESHOLD:
                if distance > vp_max[0]:
                    vp_max[0] = distance
                    vp[starting_edge:ending_edge + 1] = [0.7] * ((ending_edge + 1) - starting_edge)
                    #print("Current Max Distance DVM: ", distance)
                    test_max_2 = [vp_po, starting_edge, ending_edge, distance]
                else:
                    vp[starting_edge:ending_edge + 1] = [1] * ((ending_edge + 1) - starting_edge)
                    major_distance_count += 1
            else:
                vp[starting_edge:ending_edge + 1] = [0] * ((ending_edge + 1) - starting_edge)
                minor_distance_count +=1

            starting_edge = -1
            ending_edge = -1
        else:
            vp[position] = 0
    loop_count_vp += 1


print("majors: ", major_distance_count)
print("minors: ", minor_distance_count)

filteres_2_transposed = np.array(filtered_results_2).transpose().tolist() #untransposing to make it look good again.

#comabine horizontal and vertical
for position_main, pixel_main in enumerate(filtered_result):
    for position_idv_pix, pixel_idv_pix in enumerate(pixel_main):
        if pixel_idv_pix == 0:
            if filteres_2_transposed[position_main][position_idv_pix] == 1:
                pixel_main[position_idv_pix] = 1
            elif filteres_2_transposed[position_main][position_idv_pix] == 0.7:
                pixel_main[position_idv_pix] = 0.7
        elif filteres_2_transposed[position_main][position_idv_pix] == 0.7:
            pixel_main[position_idv_pix]= 0.7

"""END OF PIXEL COUNTING ALGORITHIM SECTION"""

"""START FINAL RESULTS"""

#APM
f_height, f_width = result_rgb.shape[0], result_rgb.shape[1]
final = cv2.resize(image,(f_width, f_height))
final[test_max[0]][test_max[1]:test_max[2]] = 1

apm = test_max[3]*PIXEL_VALUE_TO_ACTUAL_VALUE_FACTOR
print("This Oyster's APM is: " + str(apm) + "CM")
#DVM
for_dvm = np.array(final).transpose()
for_dvm[test_max_2[0]][test_max_2[1]:test_max_2[2]] = 1

dvm = test_max_2[3]*PIXEL_VALUE_TO_ACTUAL_VALUE_FACTOR
print("This Oyster's DVM is: " + str(dvm) + "CM")

actual_final = np.array(for_dvm).transpose().tolist()
plt.imshow(actual_final)

# printing out results
#plt.imshow(oysters)
#plt.imshow(filtered_results_2)
#plt.imshow(filteres_2_transposed)
#plt.imshow(filtered_result)

plt.show()
