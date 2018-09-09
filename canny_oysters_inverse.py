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
PIXEL_VALUE_TO_ACTUAL_VALUE_FACTOR = 0.04

#For pixel counting Algorithim
PIXEL_TO_LOOK = 255
INVERSE_PIXEL_TO_LOOK = 0
DISTANCE_THRESHOLD = 40
PIXEL_IGNORE_THRESHOLD = 30

#For Lens Correction
cam_maker = 'GoPro'
cam_model = 'HERO4 Silver'
lens_maker = 'GoPro'
lens_model = 'HERO4'
focal_length = 3
apperture = 2.97

#For Cropping
left_crop = 1750
right_crop = 2500
top_crop = 400
bottom_crop = 2900
NUMBER_OF_OYSTERS_HIGH = 6
NUMBER_OF_OYSTERS_WIDE = 2
# Note built for a max of 2 wide, if this needs to be changed for more than 2 code in the ROI section deeds to be edited

"""Actual Code Now"""


"""PRE-PROCESSING SECTION"""
#Reading Image
raw_image = cv2.imread('OysterImages/CustomFinal/GOPR0005.JPG')
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



oyster_in_question = '0a'.upper()
#show an image
plt.imshow(separated_oyster_images[oyster_in_question])
plt.show()
tf.reset_default_graph()

"""END OF PRE-PROCESSING SECTION"""

"""FOR THE PURPOSE OF RUNNING WHILE TESTING
SPECIFYING 1 OYSTER HERE"""

image = separated_oyster_images[oyster_in_question]

"""CANNY EDGE DETECTION SECTION"""
# Write the kernel weights as a 2D array.
filtered_result = cv2.Canny(image,10,100)
print(filtered_result)
filtered_results_2 = np.array(filtered_result).transpose().tolist()
# plt.imshow(filtered_result)
# plt.show()
"""END OF FILTERING EDGES SECTION"""


"""PIXEL COUNTING ALGORITHIM SECTION"""
major_distance_count = 0
minor_distance_count = 0

plt.imshow(filtered_result)
plt.show()

# coutning horizontal distances
hp_max = [0] #[distance, outerlistposition, start_innerlist, end_innerlist]
test_max = [0,0,0,0]
loop_count = 0
for po, hp in enumerate(filtered_result):
    starting_edge = -1
    ending_edge = -1
    live_position = -1

    for position, pixel_value in enumerate(hp[:-1]):
        if pixel_value == PIXEL_TO_LOOK and starting_edge == -1 and position not in [0,1,2,3,4]:
            starting_edge = position

        elif pixel_value == PIXEL_TO_LOOK and starting_edge != -1:
            live_position = position

        elif position == len(hp[:-1]) - 1:
            ending_edge = live_position
            distance = ending_edge - starting_edge
            if distance > DISTANCE_THRESHOLD:
                hp[starting_edge:ending_edge + 1] = [1] * ((ending_edge + 1) - starting_edge)
                major_distance_count +=1
                if distance > hp_max[0] and loop_count < filtered_result.shape[1] - PIXEL_IGNORE_THRESHOLD:
                    hp_max[0] = distance
                    hp[starting_edge:ending_edge + 1] = [0.7] * ((ending_edge + 1) - starting_edge)
                    #print("Current Max Distance APM: ", distance)
                    test_max = [po,starting_edge, ending_edge, distance]
            else:
                hp[starting_edge:ending_edge + 1] = [0] * ((ending_edge + 1) - starting_edge)
                minor_distance_count +=1

            starting_edge = -1
            ending_edge = -1
            live_position = -1
        else:
            hp[position] = 0

    loop_count +=1


# plt.imshow(filtered_result)
# plt.show()


#counting vertical distances
vp_max = [0] #[distance, outerlistposition, start_innerlist, end_innerlist]
test_max_2 = [0,0,0,0]
loop_count_vp = 0

for vp_po, vp in enumerate(filtered_results_2):
    starting_edge = -1
    ending_edge = -1
    live_position_vp = -1

    for position, pixel_value in enumerate(vp[:-1]):
        if pixel_value == PIXEL_TO_LOOK and starting_edge == -1 and position not in [0,1,2,3,4]:
            starting_edge = position

        elif pixel_value == PIXEL_TO_LOOK and starting_edge != -1:
            live_position_vp = position

        elif position == len(vp[:-1]) - 1:
            ending_edge = live_position_vp
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
            live_position_vp = -1
        else:
            vp[position] = 0
    loop_count_vp += 1


print("majors: ", major_distance_count)
print("minors: ", minor_distance_count)

filteres_2_transposed = np.array(filtered_results_2).transpose().tolist() #untransposing to make it look good again.

#comabine horizontal and vertical
# for position_main, pixel_main in enumerate(filtered_result):
#     for position_idv_pix, pixel_idv_pix in enumerate(pixel_main):
#         if pixel_idv_pix == 0:
#             if filteres_2_transposed[position_main][position_idv_pix] == 1:
#                 pixel_main[position_idv_pix] = 1
#             elif filteres_2_transposed[position_main][position_idv_pix] == 0.7:
#                 pixel_main[position_idv_pix] = 0.7
#         elif filteres_2_transposed[position_main][position_idv_pix] == 0.7:
#             pixel_main[position_idv_pix]= 0.7

"""END OF PIXEL COUNTING ALGORITHIM SECTION"""

"""START FINAL RESULTS"""
print(test_max)
print(test_max_2)
#APM
f_height, f_width = filtered_result.shape[0], filtered_result.shape[1]
final = cv2.resize(image,(f_width, f_height))
final[test_max[0]][test_max[1]:test_max[2]] = 1
apm = test_max[3]*PIXEL_VALUE_TO_ACTUAL_VALUE_FACTOR

#DVM
for_dvm = np.array(final).transpose()
for_dvm[test_max_2[0]][test_max_2[1]:test_max_2[2]] = 1
dvm = test_max_2[3]*PIXEL_VALUE_TO_ACTUAL_VALUE_FACTOR




print("This Oyster's DVM is: " + str(dvm) + "CM" + " Or: "+ str(test_max_2[3])+ " pixels")
print("This Oyster's APM is: " + str(apm) + "CM" + " Or: "+ str(test_max[3])+ " pixels")
actual_final = np.array(for_dvm).transpose().tolist()


plt.imshow(actual_final)

# printing out results
#plt.imshow(oysters)
#plt.imshow(filtered_results_2)
#plt.imshow(filteres_2_transposed)
#plt.imshow(filtered_result)

plt.show()
