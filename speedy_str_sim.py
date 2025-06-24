import time

import numpy as np
import cProfile, pstats, io
from pstats import SortKey
import cv2
import math
import pickle
import scipy.ndimage as ndi
from structural_sim_from_scratch import (setup, generate_weights, correlate1d_x_r, correlate1d_y_r, run_math, normalize_diff, correlate1d_x, correlate1d_y)
from skimage.metrics import structural_similarity


def tester(matr_to_check, correct_matrix):
    bool_arr = matr_to_check[:, :] == correct_matrix[:, :]
    val = len(bool_arr[bool_arr == False])

    if val > 0:
        arr = np.argwhere(bool_arr == False)

        for posn in arr:
            n = correct_matrix[posn[0], posn[1]]
            nd = matr_to_check[posn[0], posn[1]]
            if abs(n - nd) > 0.1:
                print(n, nd, abs(n - nd))

def find_centroid_of_contour(contour):
    """Given a contour, finds centroid of it."""
    M = cv2.moments(contour)

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy

def convert_to_contours(cell_filename):
    if isinstance(cell_filename, str):
        with open(cell_filename, 'rb') as f:
            rois = pickle.load(f)

    #            if isinstance(rois, RoiPoly):
    #                rois_dup = []
    #                for roi in rois:
    #                    rois_dup.append(roi.lines)
    #                rois = rois_dup
    else:
        rois = cell_filename

    centers = []
    contours = []

    for roi in rois:
        contour = np.array(roi, dtype='int')
        contours.append(contour)

        cx, cy = find_centroid_of_contour(contour)

        for center in centers:
            if math.dist(center[0], (cx, cy)) == 0.0:
                break
        else:
            centers.append([[cx, cy], contour])

    centers.sort(key=lambda tup: tup[0][1])

    row = 0
    col = 0
    reorg_centers = []
    curr_row = []

    # sorting by row  ------------------------
    for i in range(0, len(centers)):
        if centers[i][0][1] - centers[i - 1][0][1] > 30:
            row += 1
            curr_row.sort(key=lambda tup: tup[0][0])
            reorg_centers.append(curr_row.copy())
            curr_row.clear()

        curr_row.append(centers[i])

        if i == len(centers) - 1:
            curr_row.sort(key=lambda tup: tup[0][0])
            reorg_centers.append(curr_row)

    num_rows = len(reorg_centers)
    # ------------------------
    # TODO if not possible try to group by vertical alignment

    cell_contours = [[] for i in range(len(centers))]
    cell_centers = [[] for j in range(num_rows)]

    print(reorg_centers)
    shape_of_rows = []

    for row in range(num_rows):
        num_cols = len(reorg_centers[row])
        shape_of_rows.append(num_cols)
        for col in range(num_cols):
            y_count = row * num_cols + col

            cell_centers[row].append(reorg_centers[row][col][0])
            cell_contours[y_count] = reorg_centers[row][col][1]

    return cell_contours, cell_centers, shape_of_rows

def get_contour_mask(cell_contours, frame_width, frame_height):
    contour_mask = np.zeros((frame_height, frame_width, 3))

    for c in cell_contours:
        print(c)
        contour_mask = cv2.drawContours(contour_mask, [c],
                                        -1, (255, 255, 255), thickness=cv2.FILLED)

    contour_mask = cv2.cvtColor(
        np.array(contour_mask, dtype=np.uint8), cv2.COLOR_BGR2GRAY)

    return contour_mask

def run_correlate(curr_img, mode_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, uy_tmp, uy, uyy_tmp, uyy, width, height):
    update_corr(curr_img, mode_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, width, height)
    run_mode(mode_img, uy_tmp, uy, uyy_tmp, uyy, width, height)

def update_corr(curr_img, mode_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, width, height):
    correlate1d_x(curr_img, weights, width, height, ux_tmp)
    correlate1d_y(ux_tmp, weights, width, height, ux)
    correlate1d_x(curr_img * curr_img, weights, width, height, uxx_tmp)
    correlate1d_y(uxx_tmp, weights, width, height, uxx)
    correlate1d_x(curr_img * mode_img, weights, width, height, uxy_tmp)
    correlate1d_y(uxy_tmp, weights, width, height, uxy)

def run_mode(mode_img, uy_tmp, uy, uyy_tmp, uyy, width, height):
    correlate1d_x(mode_img, weights, width, height, uy_tmp)
    correlate1d_y(uy_tmp, weights, width, height, uy)
    correlate1d_x(mode_img * mode_img, weights, width, height, uyy_tmp)
    correlate1d_y(uyy_tmp, weights, width, height, uyy)

def run_correlate_rearr(curr_img, mode_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, uy_tmp, uy, uyy_tmp, uyy, size1, size2, width, height, cov_norm, data_range):
    run_mode_rearr(mode_img, uy_tmp, uy, uyy_tmp, uyy, size1, size2, width, height)
    update_corr_rearr(curr_img, mode_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, size1, size2, width, height)
    S = run_math(cov_norm, data_range, ux, uy, uxx, uyy, uxy)

    diff_s = normalize_diff(S, width, height)

    return diff_s

def run_mode_rearr(mode_img, uy_tmp, uy, uyy_tmp, uyy, size1, size2, width, height):
    rearr = np.concatenate((mode_img[0:size1][::-1], mode_img, mode_img[-size2:][::-1]))
    correlate1d_x_r(rearr, np_weights, uy_tmp, width, height)
    correlate1d_y(uy_tmp, np_weights, width, height, uy)

    inp = mode_img * mode_img
    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
    correlate1d_x_r(rearr, np_weights, uyy_tmp, width, height)
    correlate1d_y(uyy_tmp, np_weights, width, height, uyy)

def update_corr_rearr(curr_img, mode_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, size1, size2, width, height):
    rearr = np.concatenate((curr_img[0:size1][::-1], curr_img, curr_img[-size2:][::-1]))
    correlate1d_x_r(rearr, np_weights, ux_tmp, width, height)
    correlate1d_y(ux_tmp, np_weights, width, height, ux)
    inp = curr_img * curr_img
    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
    correlate1d_x_r(rearr, np_weights, uxx_tmp, width, height)
    correlate1d_y(uxx_tmp, np_weights, width, height, uxx)
    inp = curr_img * mode_img
    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
    correlate1d_x_r(rearr, np_weights, uxy_tmp, width, height)
    correlate1d_y(uxy_tmp, np_weights, width, height, uxy)

def run_correlate_rearr_y(curr_img, mode_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, uy_tmp, uy, uyy_tmp, uyy, size1,
                        size2, width, height, cov_norm, data_range):
    run_mode_rearr_y(mode_img, uy_tmp, uy, uyy_tmp, uyy, size1, size2, width, height)
    update_corr_rearr_y(curr_img, mode_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, size1, size2, width, height)

    S = run_math(cov_norm, data_range, ux, uy, uxx, uyy, uxy)

    S = S.transpose()
    diff_s = normalize_diff(S, width, height)

    return diff_s

def checker(og, comp1, width, height):
    comp2 = np.zeros((height, width))
    correlate1d_y(og, np_weights, width, height, comp2)

    im = np.zeros((height, width))
    im[(comp1 - comp2) > 0.001] = 1
    im[(comp1 - comp2) < -0.001] = 1

    print(len(im[im == 1]))

def run_mode_rearr_y(mode_img, uy_tmp, uy, uyy_tmp, uyy, size1, size2, width, height):
    rearr = np.concatenate((mode_img[0:size1][::-1], mode_img, mode_img[-size2:][::-1]))
    correlate1d_x_r(rearr, np_weights, uy_tmp, width, height)

    #rearr = np.concatenate((uy_tmp[:, 0:size1][:, ::-1], uy_tmp, uy_tmp[:, -size2:][:, ::-1]), axis=1) # something in the construction of this is wrong but I can't figure out what
    correlate1d_y_r(uy_tmp, np_weights, width, height, uy)

#    print("1")
#    checker(uy_tmp, uy.transpose(), width, height)
    #checker(uy.transpose(), comp_uy, width, height)

    inp = mode_img * mode_img
    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
    correlate1d_x_r(rearr, np_weights, uyy_tmp, width, height)

    #rearr = np.concatenate((uyy_tmp[:, 0:size1][:, ::-1], uyy_tmp, uyy_tmp[:, -size2:][:, ::-1]), axis=1) # something in the construction of this is wrong but I can't figure out what
    correlate1d_y_r(uyy_tmp, np_weights, width, height, uyy)
    #checker(uyy.transpose(), comp_uy, width, height)
#    print('2')
#    checker(uyy_tmp, uyy.transpose(), width, height)
#    print("------")

def update_corr_rearr_y(curr_img, mode_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, size1, size2, width, height):
    rearr = np.concatenate((curr_img[0:size1][::-1], curr_img, curr_img[-size2:][::-1]))
    correlate1d_x_r(rearr, np_weights, ux_tmp, width, height)

    #rearr = np.concatenate((ux_tmp[:, 0:size1][:, ::-1], ux_tmp, ux_tmp[:, -size2:][:, ::-1]), axis=1) # something in the construction of this is wrong but I can't figure out what
    correlate1d_y_r(ux_tmp, np_weights, width, height, ux)
#    print("3")
#    checker(ux_tmp, ux.transpose(), width, height)

    inp = curr_img * curr_img
    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
    correlate1d_x_r(rearr, np_weights, uxx_tmp, width, height)

    #rearr = np.concatenate((uxx_tmp[:, 0:size1][:, ::-1], uxx_tmp, uxx_tmp[:, -size2:][:, ::-1]), axis=1) # something in the construction of this is wrong but I can't figure out what
    correlate1d_y_r(uxx_tmp, np_weights, width, height, uxx)
#    print("4")
#    checker(uxx_tmp, uxx.transpose(), width, height)

    inp = curr_img * mode_img
    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
    correlate1d_x_r(rearr, np_weights, uxy_tmp, width, height)

    #rearr = np.concatenate((uxy_tmp[:, 0:size1][:, ::-1], uxy_tmp, uxy_tmp[:, -size2:][:, ::-1]), axis=1) # something in the construction of this is wrong but I can't figure out what
    correlate1d_y_r(uxy_tmp, np_weights, width, height, uxy)
#    print('5')
#    checker(uxy_tmp, uxy.transpose(), width, height)


def vid_runner(vidcap, mode_img, weights, data_range):
    cont, curr_img = vidcap.read()
    curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

    curr_img = curr_img.astype(np.float64, copy=False)
    mode_img = mode_img.astype(np.float64, copy=False)
    #print(curr_img.shape)
    sigma = 1.5
    truncate = 3.5
    r = int(truncate * sigma + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    ndim = curr_img.ndim
    NP = win_size ** ndim
    cov_norm = NP / (NP - 1)  # sample covariance

    C1 = (0.01 * data_range) ** 2 #K = 0.01
    C2 = (0.03 * data_range) ** 2 #K = 0.03

    ux, uy, uxx, uyy, uxy = setup(992,660) # doing this so we can transpose it later, numba requires C major order s.t. when we go in the Y direction, we want to
    # instead treat it like going in the X direction instead
    ux_tmp, uy_tmp, uxx_tmp, uyy_tmp, uxy_tmp = setup(992, 660)
    ux_t, uy_t, uxx_t, uyy_t, uxy_t = setup(660, 992)

    #correlate1d_x(curr_img, weights, ux_tmp)  # , curr_scipy)
    #correlate1d_y(curr_img, weights, ux)  # , curr_scipy)

    #S = run_math(ux, uy, uxx, uyy, uxy)

    cell_contours, cell_centers, shape_of_rows = convert_to_contours("/home/chamomile/Downloads/labview-comp/zebrafish-tracker-6-3-25.cells")
    contour_mask = get_contour_mask(cell_contours, 992, 660)

    masked_mode_noblur_img = cv2.bitwise_and(
        mode_img, mode_img, mask=contour_mask)
    masked_mode_noblur_img = mode_img.astype(np.float64, copy=False)

    height, width = (660, 992)
    weight_size = len(weights)
    size1 = math.floor(weight_size / 2)
    size2 = weight_size - size1 - 1

    run_correlate_rearr(curr_img, mode_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, uy_tmp, uy, uyy_tmp, uyy, size1,
                        size2, width, height, cov_norm, data_range)
    #run_correlate(curr_img, masked_mode_noblur_img, ux_tmp, ux_t, uxx_tmp, uxx_t, uxy_tmp, uxy_t, uy_tmp, uy_t,
    #                  uyy_tmp, uyy_t, width, height)

    run_correlate_rearr_y(curr_img, masked_mode_noblur_img, ux_tmp, ux_t, uxx_tmp, uxx_t, uxy_tmp, uxy_t, uy_tmp,
                          uy_t, uyy_tmp, uyy_t, size1,
                          size2, width, height, cov_norm, data_range)

    #np_weights = np.array(weights, dtype=np.float64)

    frame_count = 0
    tottime = 0

    pr = cProfile.Profile()
    pr.enable()
    while cont and frame_count < 100:
        #pr = cProfile.Profile()
        #pr.enable()

        t1 = time.time()

        masked_curr_img = cv2.bitwise_and(
            curr_img, curr_img, mask=contour_mask)
        masked_curr_img = masked_curr_img.astype(np.float64, copy=False)

        run_correlate_rearr(masked_curr_img, masked_mode_noblur_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, uy_tmp, uy, uyy_tmp, uyy, size1,
                            size2, width, height, cov_norm, data_range)


        diff_s = run_correlate_rearr_y(masked_curr_img, masked_mode_noblur_img, ux_tmp, ux_t, uxx_tmp, uxx_t, uxy_tmp, uxy_t, uy_tmp, uy_t, uyy_tmp, uyy_t, size1,
                              size2, width, height, cov_norm, data_range)

        cv2.imshow('diff', diff_s)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

        cont, curr_img = vidcap.read()

        curr_img_store = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        curr_img = curr_img_store.astype(np.float64, copy=False)

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
    print(tottime / frame_count)

if __name__ == '__main__':
    filename = "/home/chamomile/Downloads/labview-comp/9_20_0-392-long.avi"
    mode_noblur_path = filename[:-4] + "-mode.png"
    mode_noblur_img = cv2.cvtColor(cv2.imread(mode_noblur_path), cv2.COLOR_BGR2GRAY)

    #weights = [1, 1, 1]
    #np_weights = np.array(weights, dtype=np.float64)

    weights = generate_weights(2, sigma=1.5, truncate=3.5)[0].tolist()
    np_weights = np.array(weights, dtype=np.float64)

    weight_size = len(weights)
    size1 = math.floor(weight_size / 2)
    size2 = weight_size - size1 - 1

#    im1 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [11, 12, 13, 14, 15], [11, 12, 13, 14, 15]], dtype=np.float64)#.ascontiguousarray()
    #im2 = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])

#    print(im1.shape)
    """
    im1 = mode_noblur_img#np.random.rand(width, height)

    width, height = im1.shape[1], im1.shape[0]
    ux = np.ascontiguousarray(np.zeros((width, height)))
    ut = np.ascontiguousarray(np.zeros((height, width)))
    #    rearr = np.concatenate((im1[0:size1][::-1], im1, im1[-size2:][::-1]))

    print("im", im1)
    print("true",ndi.correlate1d(im1, weights, axis=1))
    cv2.imshow('ndi', ndi.correlate1d(im1, weights, axis=1))

    correlate1d_y_r(im1, np_weights, width, height, ux)

    print("ux", ux.transpose())
    #ux = ux.transpose()
    #print("rearr", ux.transpose())

    correlate1d_y(im1, np_weights, width, height, ut)
    print("normal", ut)
    #print(weights)
    cv2.imshow('ux', ux)
    cv2.waitKey(0)
    """

    vidcap = cv2.VideoCapture(filename)

    vid_runner(vidcap, mode_noblur_img, weights, 255)

    #vid_runner_(vidcap, mode_noblur_img, weights, 255)
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