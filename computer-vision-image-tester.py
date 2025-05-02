#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from skimage.transform import resize
from helpers import find_all_files

# num in each row
DESIRED_NUM_COLS = 8
DESIRED_NUM_ROWS = 9
fp = "/home/chamomile/Thyme-lab/data/all-cells/2_by_3/"

files_to_read = find_all_files(fp, ".png", ["contours", "BW"])
print(files_to_read)
image = cv2.imread(files_to_read[0])
width, height, _ = image.shape
row_count = 0

vert = np.zeros((0,height,3))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

for row in range(DESIRED_NUM_ROWS):
    if row < int(len(files_to_read)/DESIRED_NUM_COLS):
        hor = np.zeros((int(width/DESIRED_NUM_COLS),0,3))
        indic = (row_count*DESIRED_NUM_COLS, row_count*DESIRED_NUM_COLS + DESIRED_NUM_COLS)
        
        for filename in files_to_read[indic[0]: indic[1]]:
            image = cv2.imread(filename)
            
            # run whatever computer vision you want to test
            # ...
             
            fgmask = fgbg.apply(image)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
            cv2.imshow(f'f{filename}', fgmask)
            cv2.waitKey(0)
            # resize so all images fit 
            #image = resize(fgmask, (int(width/DESIRED_NUM_COLS), int(height/DESIRED_NUM_COLS), 3))
            
            # concat horizontally 
            #hor = np.concatenate((hor, image), axis=1)
        
        # concat vertically
        #vert = np.concatenate((vert, hor), axis=0)
        
        row_count += 1
    else:
        break
    
#cv2.imshow('f', vert)
#cv2.waitKey(0)