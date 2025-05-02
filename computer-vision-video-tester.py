#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from helpers import find_all_files, get_stats
from structural_sim_from_scratch import structural_similarity
#from skimage.metrics import structural_similarity
import numpy as np
from scipy import signal
import cProfile, pstats, io
from pstats import SortKey
import math

fp = "/home/chamomile/Thyme-lab/data/shortened_vids/6dpf/"

files_to_read = find_all_files(fp, ".avi", ["tracked"])
print(files_to_read)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

def fg(curr_img):
    fgmask = fgbg.apply(curr_img)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    return fgmask

@get_stats
def str_sim(curr_img, comp_img, weights):
    diff = structural_similarity(curr_img, comp_img, data_range=255, weights=weights)
    diff = (diff * 255).astype("uint8")
    
    cv2.imshow("diff", diff)
    thresh = cv2.threshold(diff, 150, 255, cv2.THRESH_BINARY)[1]
    return thresh

""" worse performance and slower - think convolutions aren't being done right
@get_stats
def cal_ssim(img1, img2):
    K = [0.01, 0.03]
    L = 255
    kernelX = cv2.getGaussianKernel(11, 1.5)
    window = kernelX * kernelX.T
     
    M,N = np.shape(img1)

    C1 = (K[0]*L)**2
    C2 = (K[1]*L)**2
    img1 = np.float64(img1)
    img2 = np.float64(img2)
 
    mu1 = signal.convolve2d(img1, window, 'valid')
    mu2 = signal.convolve2d(img2, window, 'valid')
    
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    
    sigma1_sq = signal.convolve2d(img1*img1, window, 'valid') - mu1_sq
    sigma2_sq = signal.convolve2d(img2*img2, window, 'valid') - mu2_sq
    sigma12 = signal.convolve2d(img1*img2, window, 'valid') - mu1_mu2
   
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    diff = (ssim_map * 255).astype("uint8")
    thresh = cv2.threshold(diff, 150, 255, cv2.THRESH_BINARY)[1]

    mssim = np.mean(ssim_map)
    cv2.imshow('mssim', mssim)
    cv2.waitKey(0)
    return thresh
"""
def run(filename):
    vidcap = cv2.VideoCapture(filename)
    cont, curr_img = vidcap.read()    
    curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

    mode_noblur_path = filename[:-4] + "-mode.png"
    mode_noblur_img = cv2.cvtColor(cv2.imread(mode_noblur_path), cv2.COLOR_BGR2GRAY)
    #mode_noblur_img[:int(mode_noblur_img.shape[0]/2),:] = 0
    #print(mode_noblur_img.shape)

    weights = [0.00102838, 0.00759876, 0.03600077, 0.10936069, 0.21300554, 0.26601172,
               0.21300554, 0.10936069, 0.03600077, 0.00759876, 0.00102838]

    np_weights = np.asarray(weights)
    weight_size = len(weights)
    size1 = math.floor(weight_size / 2)
    size2 = weight_size - size1 - 1

    pr = cProfile.Profile()
    pr.enable()
    frame_count = 0
    while frame_count < 30*60 or cont == False:
        #binary_mask = fg(curr_img)
        #print(curr_img.dtype, mode_noblur_img.dtype)



        binary_mask = str_sim(curr_img, mode_noblur_img, np_weights)

        scipy_contours = cv2.findContours(binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        scipy_contours = scipy_contours[0] if len(scipy_contours) == 2 else scipy_contours[1]

        #binary_mask = cal_ssim(curr_img, mode_noblur_img)
        
        #contours = cv2.findContours(binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        #contours = contours[0] if len(contours) == 2 else contours[1]
        
        if len(scipy_contours) > 0:
            cv2.drawContours(curr_img, scipy_contours, -1, (0,255,0), 1)
            #cv2.drawContours(curr_img, scipy_contours, -1, (255,0,0),1)
            cv2.imshow(f'f{filename}', curr_img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        cont, curr_img = vidcap.read()    
        curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        print(curr_img.dtype)
        #curr_img[:int(mode_noblur_img.shape[0]/2),:] = 0
        frame_count += 1

    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

#for filename in files_to_read:
run(files_to_read[0])

