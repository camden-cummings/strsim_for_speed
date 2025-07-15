import numpy as np
import cv2
from structural_sim_from_scratch import (run_math_complete, normalize_diff, correlate1d_x, correlate1d_y)

def run_correlate_rearr_y(curr_img, mode_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, uy_tmp, uy, uyy_tmp, uyy, size1,
                        size2, width, height, cov_norm, data_range, np_weights, weight_size):

    run_mode_rearr_y(mode_img, uy_tmp, uy, uyy_tmp, uyy, size1, size2, width, height, np_weights, weight_size)
    update_corr_rearr_y(curr_img, mode_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, size1, size2, width, height, np_weights, weight_size)

    S = run_math_complete(cov_norm, data_range, ux, uy, uxx, uyy, uxy)
    S = S.transpose()
    diff_s = normalize_diff(S, width, height)

    return diff_s

def check_similarity(im1, im2, width, height):
    im = np.zeros((height, width))
    im[(im1 - im2) > 0.001] = 1
    im[(im1 - im2) < -0.001] = 1

    return True if len(im[im == 1]) == 0 else False

def run_mode_rearr_y(mode_img, uy_tmp, uy, uyy_tmp, uyy, size1, size2, width, height, np_weights, weight_size):
    rearr = np.concatenate((mode_img[0:size1][::-1], mode_img, mode_img[-size2:][::-1]))
    correlate1d_x(rearr, np_weights, weight_size, uy_tmp, height)

    T = uy_tmp.transpose()
    rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)
    correlate1d_y(rearr, np_weights, weight_size, width, uy)
#    checker(uy_tmp, uy.transpose(), width, height)

    inp = mode_img * mode_img
    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
    correlate1d_x(rearr, np_weights, weight_size, uyy_tmp, height)

    T = uyy_tmp.transpose()
    rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)
    correlate1d_y(rearr, np_weights, weight_size, width, uyy)
#    checker(uyy_tmp, uyy.transpose(), width, height)


def update_corr_rearr_y(curr_img, mode_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, size1, size2, width, height, np_weights, weight_size):
    rearr = np.concatenate((curr_img[0:size1][::-1], curr_img, curr_img[-size2:][::-1]))
    correlate1d_x(rearr, np_weights, weight_size, ux_tmp, height)

    T = ux_tmp.transpose()
    rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)
    correlate1d_y(rearr, np_weights, weight_size, width, ux)
#    checker(ux_tmp, ux.transpose(), width, height)

    inp = curr_img * curr_img
    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
    correlate1d_x(rearr, np_weights, weight_size, uxx_tmp, height)

    T = uxx_tmp.transpose()
    rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)
    correlate1d_y(rearr, np_weights, weight_size, width, uxx)
#    checker(uxx_tmp, uxx.transpose(), width, height)

    inp = curr_img * mode_img
    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
    correlate1d_x(rearr, np_weights, weight_size, uxy_tmp, height)

    T = uxy_tmp.transpose()
    rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)
    correlate1d_y(rearr, np_weights, weight_size, width, uxy)
#    checker(uxy_tmp, uxy.transpose(), width, height)

"""
if __name__ == '__main__':
    filename = "/home/chamomile/Downloads/labview-comp/9_20_0-392-long.avi"
    vidcap = cv2.VideoCapture(filename)

    mode_noblur_path = filename[:-4] + "-mode.png"
    mode_noblur_img = cv2.cvtColor(cv2.imread(mode_noblur_path), cv2.COLOR_BGR2GRAY)

    weights = generate_weights(2, sigma=1.5, truncate=3.5)[0].tolist()
    np_weights = np.array(weights)

"""



"""
    #vid_runner_(vidcap, mode_noblur_img, np_weights, 255)
    vid_runner(vidcap, mode_noblur_img, weights, 255, 992, 660)
"""
