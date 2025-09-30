import cProfile
import io
import math
import os
import pstats
import time
from pstats import SortKey
import copy

import cv2
import numpy as np

from computer_vision.speedy_str_sim import run_correlate_rearr_y, run_correlate_rearr_y__, normalize_diff, update_corr_rearr_y, run_math_complete
from computer_vision.structural_sim_from_scratch import setup, generate_weights
from computer_vision.profile_wrap import start_profiler, end_profiler
from skimage.metrics import structural_similarity as ssim

def compare_methods_on_vid(vidcap, weights, data_range, frame_width, frame_height, cov_norm):
    cont, curr_img = vidcap.read()
    curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

    curr_img = curr_img.astype(np.float32, copy=False, order='C')

    ux_tmp, uy_tmp, uxx_tmp, uyy_tmp, uxy_tmp = setup(frame_width, frame_height, 'C', np.float32)
    ux, uy, uxx, uyy, uxy = setup(frame_height, frame_width, 'C', np.float32)

    ux_tmp__, uy_tmp__, uxx_tmp__, uyy_tmp__, uxy_tmp__ = setup(frame_height, frame_width, 'C', np.float32)
    ux__, uy__, uxx__, uyy__, uxy__ = setup(frame_height, frame_width, 'C', np.float32)

    weight_size = len(weights)
    size1 = math.floor(weight_size / 2)
    size2 = weight_size - size1 - 1
    np_weights = np.array(weights, dtype=np.float32, order='C')

    out = np.zeros((frame_height, frame_width), dtype=np.uint8)
    prev_curr_img = np.zeros((frame_height, frame_width), dtype=np.float32, order='C')

    run_correlate_rearr_y(curr_img, prev_curr_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, uy_tmp,
                          uy, uyy_tmp, uyy, size1,
                          size2, frame_width, frame_height, cov_norm, data_range, np_weights, weight_size, out)

    run_correlate_rearr_y__(curr_img, prev_curr_img, ux_tmp__, ux__, uxx_tmp__, uxx__, uxy_tmp__, uxy__, uy_tmp__,
                            uy__, uyy_tmp__, uyy__, size1,
                            size2, frame_width, frame_height, cov_norm, data_range, np_weights, weight_size, out)

    frame_count = 0
    prev_curr_img = np.zeros((frame_height, frame_width), dtype=np.float32, order='C')
    #pr = start_profiler()
    """
    while cont and frame_count < 100:
        curr_img = curr_img.astype(np.float32, copy=False, order='C')

        pr = start_profiler()

        score, ssim_const = ssim(curr_img, prev_curr_img, data_range=255, full=True, gaussian_weights=True)
        normalize_diff(ssim_const, frame_width, frame_height, out)

        prev_curr_img = curr_img
        end_profiler(pr)

        cont, curr_img = vidcap.read()
        curr_img_store = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        curr_img = curr_img_store.astype(np.float32, copy=False)

        frame_count += 1

    #end_profiler(pr)
    """

    frame_count = 0
    prev_curr_img = np.zeros((frame_height, frame_width), dtype=np.float32, order='C')
    pr = start_profiler()
    while cont and frame_count < 100:
        curr_img = curr_img.astype(np.float32, copy=False, order='C')

        #pr = start_profiler()

        run_correlate_rearr_y(curr_img, prev_curr_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, uy_tmp,
                              uy, uyy_tmp, uyy, size1,
                              size2, frame_width, frame_height, cov_norm, data_range, np_weights, weight_size, out)

        contours = run_correlate_rearr_y__(curr_img, prev_curr_img, ux_tmp__, ux__, uxx_tmp__, uxx__, uxy_tmp__, uxy__, uy_tmp__,
                              uy__, uyy_tmp__, uyy__, size1,
                              size2, frame_width, frame_height, cov_norm, data_range, np_weights, weight_size, out)

        out_to_colour_constrained_centroids = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

        cv2.drawContours(out_to_colour_constrained_centroids, contours, -1, (0, 0, 255), 1)

        #cv2.imshow('f',out_to_colour_constrained_centroids)
        #cv2.waitKey(0)
        prev_curr_img = curr_img
        #end_profiler(pr)

        cont, curr_img = vidcap.read()
        curr_img_store = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        curr_img = curr_img_store.astype(np.float32, copy=False)

        frame_count += 1

    end_profiler(pr)



    """
    frame_count = 0
    prev_curr_img = np.zeros((frame_height, frame_width), dtype=np.float32, order='C')
    #pr = start_profiler()
    while cont and frame_count < 100:
        curr_img = curr_img.astype(np.float32, copy=False, order='C')

        #pr = start_profiler()

        update_corr_rearr_y(curr_img, prev_curr_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, size1, size2, frame_width, frame_height,
                            np_weights, weight_size)

        bufferarr = run_math_complete(cov_norm, data_range, ux, uy, uxx, uyy, uxy)
        S_t = bufferarr.T
        normalize_diff(S_t, frame_width, frame_height, out)

        uy = ux.copy()
        uyy = uxx.copy()

        prev_curr_img = curr_img
        #end_profiler(pr)

        cont, curr_img = vidcap.read()
        curr_img_store = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        curr_img = curr_img_store.astype(np.float32, copy=False)

        frame_count += 1
    """
#    end_profiler(pr)


if __name__ == '__main__':
    filename = "/home/chamomile/Thyme_lab/data/combined_vids_csvs/10dpf/2024-02-16_1run_2024-02-16-112634-0000.avi"

    vidcap = cv2.VideoCapture(filename)

    frame_width = int(vidcap.get(3))
    frame_height = int(vidcap.get(4))

    weights, cov_norm = generate_weights(ndim=2, sigma=1.5, truncate=3.5)

    weights = weights.tolist()

    cont, curr_img = vidcap.read()

    compare_methods_on_vid(vidcap, weights, 255, frame_width, frame_height, cov_norm)
