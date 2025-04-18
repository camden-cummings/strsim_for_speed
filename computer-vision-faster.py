#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from skimage.transform import resize

from helpers.file_manip import find_all_videos_for_tracking

# num in each row
DESIRED_NUM_COLS = 8
DESIRED_NUM_ROWS = 9
fp = "/path/to/image/folder"

files_to_read = find_all_videos_for_tracking(fp, ".png")
image = cv2.imread(files_to_read[0])
width, height, _ = image.shape
row_count = 0

vert = np.zeros((0,height,3))

for row in range(DESIRED_NUM_ROWS):
    if row < int(len(files_to_read)/DESIRED_NUM_COLS):
        hor = np.zeros((int(width/DESIRED_NUM_COLS),0,3))
        indic = (row_count*DESIRED_NUM_COLS, row_count*DESIRED_NUM_COLS + DESIRED_NUM_COLS)
        
        for filename in files_to_read[indic[0]: indic[1]]:
            image = cv2.imread(filename)
            
            # run whatever computer vision you want to test
            # ...
            
            # resize so all images fit 
            image = resize(image, (int(width/DESIRED_NUM_COLS), int(height/DESIRED_NUM_COLS), 3))
            
            # concat horizontally 
            hor = np.concatenate((hor, image), axis=1)
        
        # concat vertically
        vert = np.concatenate((vert, hor), axis=0)
        
        row_count += 1
    else:
        break
    
cv2.imshow('f', vert)
cv2.waitKey(0)