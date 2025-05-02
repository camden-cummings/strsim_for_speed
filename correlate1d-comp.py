import os
#os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
from helpers import find_all_files
import cv2
from structural_sim_from_scratch import vid_runner, structural_similarity, setup, run_math, generate_weights, correlate1d
import cProfile, pstats, io
from pstats import SortKey

np.random.seed(1)
im1 = np.random.randint(1, 1000, size=(1200,1760), dtype=int)
im2 = np.random.randint(1, 1000, size=(1200,1760), dtype=int)

weights = [0.00102838, 0.00759876, 0.03600077, 0.10936069, 0.21300554, 0.26601172,
 0.21300554, 0.10936069, 0.03600077, 0.00759876, 0.00102838]

np_weights = np.asarray(weights)

fp = "/home/chamomile/Thyme-lab/data/shortened_vids/6dpf/"

files_to_read = find_all_files(fp, ".avi", ["tracked"])

filename = files_to_read[0]
mode_noblur_path = filename[:-4] + "-mode.png"
mode_noblur_img = cv2.cvtColor(cv2.imread(mode_noblur_path), cv2.COLOR_BGR2GRAY)

vidcap = cv2.VideoCapture(filename)

vid_runner(vidcap, mode_noblur_img, np_weights, 255)

#diff = structural_similarity(im, im2, 255, weights)

"""
weights, cov_norm = generate_weights(im1.ndim)
ux, uy, uxx, uyy, uxy = setup(im1, im2, weights)
S = run_math(cov_norm, 255, ux, uy, uxx, uyy, uxy)

#output = np.zeros((1200,1760))
pr = cProfile.Profile()
pr.enable()

for i in range(50):
    #diff = structural_similarity(im, im2, 255, weights)
    correlate1d(im1, weights, ux)
    correlate1d(im2, weights, uy)

    correlate1d(im1 * im1, weights, uxx)
    correlate1d(im2 * im2, weights, uyy)
    correlate1d(im1 * im2, weights, uxy)

    run_math(cov_norm, 255, ux, uy, uxx, uyy, uxy)

pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
"""