"""
Script to compute the tubule interstism ratio for given masks
"""


import cv2
import os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

data_dir = r'kidney_mif/masks'

core_biopsy_mask = np.load(r'kidney_mif/masks/213_HIVE3_TMA_191_7_10_6_22_Scan2-v3/core_biopsy.npy')

tubules = np.load(r'kidney_mif/masks/213_HIVE3_TMA_191_7_10_6_22_Scan2-v3/capillary_seg.npy')


core_biopsy_mask.dtype = 'uint8'
ext_contours, heirarchy = cv2.findContours(image = core_biopsy_mask, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_NONE)

ext_contours = list(ext_contours)
ext_contours.sort(key=len, reverse = True)
num_biopsies = 6
mif_aspect_ratio = 0.5077

ratios = []

for i in range(num_biopsies):
    # get core biopsy
    temp_tubule = tubules.copy()

    temp_mask = np.zeros_like(core_biopsy_mask)
    temp_mask.dtype = 'uint8'
    cv2.drawContours(temp_mask, [ext_contours[i]], 0, 255 , -1)


    temp_tubule[temp_mask != 255] = 0

    max_val = np.max(temp_tubule)

    total_tubule = np.sum(temp_tubule/ max_val) # divide by 255 because I saved mask as float not binary

    area = cv2.contourArea(ext_contours[i])

    # CHECK UNITS
    ratio = total_tubule / area

    ratios.append(ratio)

print(ratios)
