import cProfile
import io
import math
import os
import pstats
import time
from pstats import SortKey

import cv2
import numpy as np

from helpers import get_contour_mask, calc_mode_img
from speedy_str_sim import run_correlate_rearr_y
from structural_sim_from_scratch import setup, generate_weights


def vid_runner(vidcap, mode_img, weights, data_range, frame_width, frame_height, cov_norm):
    cont, curr_img = vidcap.read()
    curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

    curr_img = curr_img.astype(np.float32, copy=False, order='C')
    mode_img = mode_img.astype(np.float32, copy=False, order='C')

    ux_tmp, uy_tmp, uxx_tmp, uyy_tmp, uxy_tmp = setup(frame_width, frame_height, 'C')
    ux, uy, uxx, uyy, uxy = setup(frame_height, frame_width, 'C')

    contour_mask = get_contour_mask("/home/chamomile/Thyme-lab/data/vids/social_and_many_well/to-be-processed/march/fc2_save_2025-03-18-142549-0000.cells", frame_width, frame_height)

    masked_mode_noblur_img = cv2.bitwise_and(
        mode_img, mode_img, mask=contour_mask)
    masked_mode_noblur_img = masked_mode_noblur_img.astype(np.float32, copy=False, order='C')

    weight_size = len(weights)
    size1 = math.floor(weight_size / 2)
    size2 = weight_size - size1 - 1
    np_weights = np.array(weights, dtype=np.float32, order='C')

    #run_correlate_rearr(curr_img, mode_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, uy_tmp, uy, uyy_tmp, uyy, size1,
    #                    size2, width, height, cov_norm, data_range)

    run_correlate_rearr_y(curr_img, masked_mode_noblur_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, uy_tmp,
                          uy, uyy_tmp, uyy, size1,
                          size2, frame_width, frame_height, cov_norm, data_range, np_weights, weight_size)

    frame_count = 0
    tottime = 0

    prev_curr_img = np.zeros((frame_height, frame_width), dtype=np.float32, order='C')
    pr = cProfile.Profile()
    pr.enable()
    while cont and frame_count < 100:
        #pr = cProfile.Profile()
        #pr.enable()

        t1 = time.time()

        masked_curr_img = cv2.bitwise_and(curr_img, curr_img, mask=contour_mask)
        masked_curr_img = masked_curr_img.astype(np.float32, copy=False, order='C')

        #run_correlate_rearr(masked_curr_img, masked_mode_noblur_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, uy_tmp, uy, uyy_tmp, uyy, size1,
        #                    size2, width, height, cov_norm, data_range)

        diff_s = run_correlate_rearr_y(masked_curr_img, prev_curr_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, uy_tmp, uy, uyy_tmp, uyy, size1,
                              size2, frame_width, frame_height, cov_norm, data_range, np_weights, weight_size)
        uy = ux.copy()
        uyy = uxx.copy()

        """cv2.imshow('diff', diff_s)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        """

        cont, curr_img = vidcap.read()
        #cv2.imshow('curr', curr_img)

        prev_curr_img = masked_curr_img

        curr_img_store = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
#        cv2.imshow('currstore', curr_img_store)

        curr_img = curr_img_store.astype(np.float32, copy=False)
#        cv2.imshow('curr?', curr_img)

        frame_count += 1
        t0 = time.time()
        total = t0 - t1
        tottime += total

        #pr.disable()
        #s = io.StringIO()
        #sortby = SortKey.CUMULATIVE
        #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        #ps.print_stats()
        #print(s.getvalue())
        #print(tottime / frame_count)

    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    #print(tottime / frame_count)


if __name__ == '__main__':
    filename = "/home/chamomile/Thyme-lab/data/vids/social_and_many_well/to-be-processed/march/fc2_save_2025-03-18-142549-0000.mp4"
    mode_noblur_path = filename[:-4] + "-mode.png"

    vidcap = cv2.VideoCapture(filename)

    frame_width = int(vidcap.get(3))
    frame_height = int(vidcap.get(4))

    if not os.path.exists(mode_noblur_path):
        calc_mode_img(vidcap, frame_width, frame_height, mode_noblur_path, False)

    mode_noblur_img = cv2.cvtColor(cv2.imread(mode_noblur_path), cv2.COLOR_BGR2GRAY)

    weights, cov_norm = generate_weights(ndim=2, sigma=1.5, truncate=3.5)
    print(cov_norm)
    weights = weights.tolist()

    vid_runner(vidcap, mode_noblur_img, weights, 255, frame_width, frame_height, cov_norm)
