import time

import numpy as np
import cProfile, pstats, io
from pstats import SortKey
import cv2

from structural_sim_from_scratch import setup, generate_weights, correlate1d_x, correlate1d_y, run_math, normalize_diff, height, width

#from skimage.metrics import structural_similarity


def run(diff):
    return (diff * 255).astype("uint8")

def tester(matr_to_check, correct_matrix):
    bool_arr = matr_to_check[:, :] == correct_matrix[:, :]
    val = len(bool_arr[bool_arr == False])

    if val > 0:
        arr = np.argwhere(bool_arr == False)

        for posn in arr:
            n = correct_matrix[posn[0], posn[1]]
            nd = matr_to_check[posn[0], posn[1]]
            if abs(n - nd) > 0.5:
                print(n, nd, abs(n - nd))

def vid_runner(vidcap, mode_img, weights, data_range):
    cont, curr_img = vidcap.read()
    curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    curr_img_store = np.zeros((height, width))

    """
    # masked image ex
    mask = np.zeros((height, width, 3))
    cv2.drawContours(mask, [np.array([[400, 100], [400,200], [100, 100], [100, 200]], dtype=np.int32)], -1, (255, 255, 255), cv2.FILLED)
    mask = cv2.cvtColor(np.array(mask, np.uint8), cv2.COLOR_BGR2GRAY)
    mask = mask.astype(np.uint8, copy=False)

    curr_img = cv2.bitwise_and(curr_img, curr_img, mask=mask)
    mode_img = cv2.bitwise_and(mode_img, mode_img, mask=mask)
    """

    curr_img = curr_img.astype(np.float64, copy=False)
    mode_img = mode_img.astype(np.float64, copy=False)

    ux, uy, uxx, uyy, uxy = setup(width,height)
    ux_tmp, uy_tmp, uxx_tmp, uyy_tmp, uxy_tmp = setup(width, height)

    correlate1d_x(mode_img * mode_img, weights, uyy_tmp)
    correlate1d_y(uyy_tmp, weights, uyy)
    correlate1d_x(mode_img, weights, uy_tmp)
    correlate1d_y(uy_tmp, weights, uy)

    sigma = 1.5
    truncate = 3.5
    r = int(truncate * sigma + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    ndim = curr_img.ndim
    NP = win_size ** ndim
    cov_norm = NP / (NP - 1)  # sample covariance

    C1 = (0.01 * data_range) ** 2 #K = 0.01
    C2 = (0.03 * data_range) ** 2 #K = 0.03

    vy = cov_norm * (uyy - uy*uy)

    tottime = 0
    frame_count = 0
    while cont and frame_count < 50:
        pr = cProfile.Profile()
        pr.enable()

        t1 = time.time()

        correlate1d_x(curr_img, weights, ux_tmp)
        correlate1d_y(ux_tmp, weights, ux)
        correlate1d_x(curr_img * curr_img, weights, uxx_tmp)
        correlate1d_y(uxx_tmp, weights, uxx)
        correlate1d_x(curr_img * mode_img, weights, uxy_tmp)
        correlate1d_y(uxy_tmp, weights, uxy)

        diff = run_math(cov_norm, data_range, ux, uy, uxx, vy, uxy)
        diff_s = normalize_diff(diff)

        t0 = time.time()

        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

        total = t0 - t1
        tottime += total
#        print(total)

#        cv2.imshow('diff', diff)

        # find contours // 
        thresh = cv2.threshold(diff, 150, 255, cv2.THRESH_BINARY)[1]

        scipy_contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        scipy_contours = scipy_contours[0] if len(scipy_contours) == 2 else scipy_contours[1]

#        sort_contours_by_area(scipy_contours, frame_count, time, diff, mask, shape_of_rows, cell_contours, cell_centers)

        if len(scipy_contours) > 0:
            cv2.drawContours(curr_img_store, scipy_contours, -1, (0,255,0), 1)
        cv2.drawContours(curr_img, scipy_contours, -1, (255,0,0),1)
        cv2.imshow('f', curr_img_store)

        cv2.imshow('f', thresh)


        """
        # compare to scikit.ndimage
        (score, diff_strsim) = structural_similarity(curr_img, mode_img, full=True, data_range=255, gaussian_weights=True)
        diff_strsim = normalize_diff(diff_strsim)

        cv2.imshow('diff_strsim', diff_strsim)
        """

        """
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        """

        cont, curr_img = vidcap.read()

        curr_img_store = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        curr_img = curr_img_store.astype(np.float64, copy=False)

        frame_count += 1


    print(tottime/51)
    #correlate1d_x.parallel_diagnostics(level=4)
    #correlate1d_y.parallel_diagnostics(level=4)
    #run_math.parallel_diagnostics(level=4)

if __name__ == '__main__':
    filename = "/home/chamomile/Thyme-lab/data/fc2_save_2025-05-08-121318-0000.mp4"
    mode_noblur_path = filename[:-4] + "-mode.png"
    mode_noblur_img = cv2.cvtColor(cv2.imread(mode_noblur_path), cv2.COLOR_BGR2GRAY)

    vidcap = cv2.VideoCapture(filename)
    weights = generate_weights(2, sigma=1.5, truncate=3.5)[0].tolist()

    vid_runner(vidcap, mode_noblur_img, weights, 255)
