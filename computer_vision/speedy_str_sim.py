import time

import numpy as np
import cProfile
import io
import math
import os
import pstats
import time
from pstats import SortKey
import copy
import cv2

from .structural_sim_from_scratch import (run_math_complete, run_math_complete_, normalize_diff, correlate1d_x_ as correlate1d_x, correlate1d_y, correlate1d_x__, correlate1d_y__)


def run_correlate_rearr_y(curr_img, mode_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, uy_tmp, uy, uyy_tmp, uyy, size1,
                        size2, width, height, cov_norm, data_range, np_weights, weight_size, out):
    run_mode_rearr_y(mode_img, uy_tmp, uy, uyy_tmp, uyy, size1, size2, width, height, np_weights, weight_size)
    update_corr_rearr_y(curr_img, mode_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, size1, size2, width, height, np_weights, weight_size)

#    print("ux", np.min(ux), np.max(ux))

    bufferarr = run_math_complete(cov_norm, data_range, ux, uy, uxx, uyy, uxy)

    S_t = bufferarr.T

    normalize_diff(S_t, width, height, out)
#    print(out)
#    print("out", np.min(out), np.max(out))

    #cv2.imshow('diff', out)

    thresh = cv2.threshold(out, 150, 255, cv2.THRESH_BINARY)[1]
    #cv2.imshow('thresh', thresh)
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

#    print(len(contours))
    out_to_colour = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    #cv2.drawContours(out_to_colour, contours, -1, (0, 0, 255), 1)
    #cv2.imshow('diff_contours', out_to_colour)

    contours = [c for c in contours if 10 < cv2.contourArea(c) < 300]
    out_to_colour_constrained_centroids = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    #cv2.drawContours(out_to_colour_constrained_centroids, contours, -1, (0, 0, 255), 1)
#    cv2.imshow('diff_less_contours', out_to_colour_constrained_centroids)

    #cv2.waitKey(0)

    #contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]

def run_correlate_rearr_y__(curr_img, mode_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, uy_tmp, uy, uyy_tmp, uyy, size1,
                        size2, width, height, cov_norm, data_range, np_weights, weight_size, out):
    run_mode_rearr_y__(mode_img, uy_tmp, uy, uyy_tmp, uyy, size1, size2, width, height, np_weights, weight_size)
    update_corr_rearr_y__(curr_img, mode_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, size1, size2, width, height, np_weights, weight_size)

    #    print("ux", np.min(ux), np.max(ux))
    bufferarr = run_math_complete(cov_norm, data_range, ux, uy, uxx, uyy, uxy)
    S_t = bufferarr.T

    normalize_diff(S_t, width, height, out)
#    print(out)
    #    print("out", np.min(out), np.max(out))

    # cv2.imshow('diff', out)

    thresh = cv2.threshold(out, 150, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow('thresh', thresh)
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    out_to_colour = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    contours = [c for c in contours if 10 < cv2.contourArea(c) < 300]
    out_to_colour_constrained_centroids = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

def check_similarity(im1, im2, width, height):
    """
    Parameters
    ----------
    im1
    im2
    width
    height

    Returns
    -------
    unknown
    """
    im = np.zeros((height, width))
    im[(im1 - im2) > 0.001] = 1
    im[(im1 - im2) < -0.001] = 1

    return True if len(im[im == 1]) == 0 else False

def run_mode_rearr_y(mode_img, uy_tmp, uy, uyy_tmp, uyy, size1, size2, width, height, np_weights, weight_size):
    rearr = np.concatenate((mode_img[0:size1][::-1], mode_img, mode_img[-size2:][::-1]))
    correlate1d_x(rearr, np_weights, weight_size, height, uy_tmp)

    T = uy_tmp.T
    rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)
    correlate1d_y(rearr, np_weights, weight_size, width, uy)

    inp = mode_img * mode_img
    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
    correlate1d_x(rearr, np_weights, weight_size, height, uyy_tmp)

    T = uyy_tmp.T
    rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)
    correlate1d_y(rearr, np_weights, weight_size, width, uyy)

def run_mode_rearr_y__(mode_img, uy_tmp, uy, uyy_tmp, uyy, size1, size2, width, height, np_weights, weight_size):
    rearr = np.concatenate((mode_img[0:size1][::-1], mode_img, mode_img[-size2:][::-1]))
    correlate1d_x__(rearr, np_weights, weight_size, height, uy_tmp)

    rearr = np.concatenate((uy_tmp[0:size1][::-1], uy_tmp, uy_tmp[-size2:][::-1]), axis=0)
    correlate1d_y__(rearr, np_weights, weight_size, width, uy)

    inp = mode_img * mode_img
    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
    correlate1d_x__(rearr, np_weights, weight_size, height, uyy_tmp)

    rearr = np.concatenate((uyy_tmp[0:size1][::-1], uyy_tmp, uyy_tmp[-size2:][::-1]), axis=0)
    correlate1d_y__(rearr, np_weights, weight_size, width, uyy)

def update_corr_rearr_y(curr_img, mode_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, size1, size2, width, height, np_weights, weight_size):
    rearr = np.concatenate((curr_img[0:size1][::-1], curr_img, curr_img[-size2:][::-1]))
    correlate1d_x(rearr, np_weights, weight_size, height, ux_tmp)

    T = ux_tmp.T
    rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)
    correlate1d_y(rearr, np_weights, weight_size, width, ux)

    inp = curr_img * curr_img
    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
    correlate1d_x(rearr, np_weights, weight_size, height, uxx_tmp)

    T = uxx_tmp.T
    rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)
    correlate1d_y(rearr, np_weights, weight_size, width, uxx)

    inp = curr_img * mode_img
    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
    correlate1d_x(rearr, np_weights, weight_size, height, uxy_tmp)

    T = uxy_tmp.T
    rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)
    correlate1d_y(rearr, np_weights, weight_size, width, uxy)

def update_corr_rearr_y__(curr_img, mode_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, size1, size2, width, height, np_weights, weight_size):
    rearr = np.concatenate((curr_img[0:size1][::-1], curr_img, curr_img[-size2:][::-1]))
    correlate1d_x__(rearr, np_weights, weight_size, height, ux_tmp)

    rearr = np.concatenate((ux_tmp[0:size1][::-1], ux_tmp, ux_tmp[-size2:][::-1]), axis=0)
    correlate1d_y__(rearr, np_weights, weight_size, width, ux)

    inp = curr_img * curr_img
    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
    correlate1d_x__(rearr, np_weights, weight_size, height, uxx_tmp)

    rearr = np.concatenate((uxx_tmp[0:size1][::-1], uxx_tmp, uxx_tmp[-size2:][::-1]), axis=0)
    correlate1d_y__(rearr, np_weights, weight_size, width, uxx)

    inp = curr_img * mode_img
    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
    correlate1d_x__(rearr, np_weights, weight_size, height, uxy_tmp)

    rearr = np.concatenate((uxy_tmp[0:size1][::-1], uxy_tmp, uxy_tmp[-size2:][::-1]), axis=0)
    correlate1d_y__(rearr, np_weights, weight_size, width, uxy)
