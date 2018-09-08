import lensfunpy
import cv2
import matplotlib.pyplot as plt

cam_maker = 'GoPro'
cam_model = 'HERO4 Silver'
lens_maker = 'GoPro'
lens_model = 'HERO4'

db = lensfunpy.Database()
cam = db.find_cameras(cam_maker, cam_model)[0]
lens = db.find_lenses(cam, lens_maker, lens_model)[0]

print(cam)
print(lens)

im = 'OysterImages/1 (20).JPG'
image = cv2.imread(im)
height, width = image.shape[0], image.shape[1]
mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)
mod.initialize(30, 2.97, 1)
#[focal length, aperture, distance]

undist_coords = mod.apply_geometry_distortion()
im_undistorted = cv2.remap(image, undist_coords, None, cv2.INTER_LANCZOS4)
plt.imshow(im_undistorted)
plt.show()