import time

import numpy as np

from .structural_sim_from_scratch import (run_math_complete, normalize_diff, correlate1d_x, correlate1d_y, correlate1d_y_wrap)


def run_correlate_rearr_y(curr_img, mode_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, uy_tmp, uy, uyy_tmp, uyy, size1,
                        size2, width, height, cov_norm, data_range, np_weights, weight_size):

    run_mode_rearr_y(mode_img, uy_tmp, uy, uyy_tmp, uyy, size1, size2, width, height, np_weights, weight_size)
    update_corr_rearr_y(curr_img, mode_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, size1, size2, width, height, np_weights, weight_size)

    S = run_math_complete(cov_norm, data_range, ux, uy, uxx, uyy, uxy)
    S = S.transpose()
    diff_s = normalize_diff(S, width, height)

    return diff_s

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

    T = uy_tmp.transpose()
    rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)
    #rearr = np.ascontiguousarray(rearr)
    correlate1d_y(rearr, np_weights, weight_size, width, uy)

#    check_similarity(uy_tmp, uy.transpose(), width, height)

    inp = mode_img * mode_img
    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
    correlate1d_x(rearr, np_weights, weight_size, height, uyy_tmp)


    T = uyy_tmp.transpose()
    rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)
    #rearr = np.ascontiguousarray(rearr)
    correlate1d_y(rearr, np_weights, weight_size, width, uyy)

#    check_similarity(uyy_tmp, uyy.transpose(), width, height)


def update_corr_rearr_y(curr_img, mode_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, size1, size2, width, height, np_weights, weight_size):
    rearr = np.concatenate((curr_img[0:size1][::-1], curr_img, curr_img[-size2:][::-1]))
    correlate1d_x(rearr, np_weights, weight_size, height, ux_tmp)

    #correlate1d_y_wrap(ux_tmp, np_weights, weight_size, width, ux, size1, size2)

    T = ux_tmp.transpose()
    rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)
    #rearr = np.ascontiguousarray(rearr)
    correlate1d_y(rearr, np_weights, weight_size, width, ux)

#    check_similarity(ux_tmp, ux.transpose(), width, height)

    inp = curr_img * curr_img
    #rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
    correlate1d_x(rearr, np_weights, weight_size, height, uxx_tmp)

    #correlate1d_y_wrap(uxx_tmp, np_weights, weight_size, width, uxx, size1, size2)

    T = uxx_tmp.transpose()
    rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)
    #rearr = np.ascontiguousarray(rearr)
    correlate1d_y(rearr, np_weights, weight_size, width, uxx)

#    check_similarity(uxx_tmp, uxx.transpose(), width, height)

    inp = curr_img * mode_img
    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
    correlate1d_x(rearr, np_weights, weight_size, height, uxy_tmp)

    #correlate1d_y_wrap(uxy_tmp, np_weights, weight_size, width, uxy, size1, size2)

    T = uxy_tmp.transpose()
    rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)
    #rearr = np.ascontiguousarray(rearr)
    correlate1d_y(rearr, np_weights, weight_size, width, uxy)

#    check_similarity(uxy_tmp, uxy.transpose(), width, height)