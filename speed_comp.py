import cProfile
import io
import math
import pstats
import time
from pstats import SortKey

import numpy as np

from comparison import correlate1d_x as correlate1d_x_, correlate1d_y as correlate1d_y_
from speedy_str_sim import run_correlate_rearr_y
from structural_sim_from_scratch import setup, generate_weights, __correlate1d_x as correlate1d_x, \
    __correlate1d_y as correlate1d_y

# compare scipy strsim, other python implementations (?) to own

#from numba import config, threading_layer

# set the threading layer before any parallel target compilation
#config.THREADING_LAYER = 'safe'

width, height = 1000, 1000

def generate_test_imageset(width, height, num_of_tests):
    np.random.seed(0)
    testset = []
    for i in range(num_of_tests):
        img1 = np.array(np.random.randint(100, size=(width, height)), dtype=np.float32)
        img2 = np.array(np.random.randint(101, size=(width, height)), dtype=np.float32)

        #img1 = np.random.randint(100, size=(width, height))
        #img2 = np.random.randint(101, size=(width, height))

        testset.append((img1, img2))

    return testset

def start_profiler():
    pr = cProfile.Profile()
    pr.enable()

    return pr

def end_profiler(pr):
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

testset = generate_test_imageset(1000, 1000, 100)
weights, cov_norm = generate_weights(ndim=2, sigma=1.5, truncate=3.5)
np_weights = np.array(weights, dtype=np.float32, order='C')
weight_size = len(weights)
size1 = math.floor(weight_size / 2)
size2 = weight_size - size1 - 1

ux_tmp, uy_tmp, uxx_tmp, uyy_tmp, uxy_tmp = setup(width, height, 'C')
ux, uy, uxx, uyy, uxy = setup(height, width, 'C')
img1 = testset[0][0]
img2 = testset[0][1]
run_correlate_rearr_y(img1, img2, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, uy_tmp,
                      uy, uyy_tmp, uyy, size1,
                      size2, width, height, cov_norm, 255, np_weights, weight_size)

profiler = start_profiler()

for img1, img2 in testset:
#    _, S_strsim = structural_similarity(img1, img2, full=True, data_range=255, gaussian_weights=True)
    run_correlate_rearr_y(img1, img2, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, uy_tmp,
                          uy, uyy_tmp, uyy, size1,
                          size2, width, height, cov_norm, 255, np_weights, weight_size)

end_profiler(profiler)

# compare correlate1d against scipy correlate1d

output = np.zeros((width, height), dtype=np.float32)

correlate1d_x_(img1, np_weights, output, width, height)
correlate1d_x(img1, np_weights, weight_size, output, height)

correlate1d_y_(img1, np_weights, output, width, height)
correlate1d_y(img1, np_weights, weight_size, width, output)

tot_1d_ = 0
tot_1d = 0

profiler = start_profiler()

for img1, img2 in testset:
    start_time = time.time()

    correlate1d_x_(img1, np_weights, output, width, height)

    end_time = time.time()
    tot_1d_ += end_time - start_time

    start_time = time.time()

    correlate1d_x(img1, np_weights, weight_size, output, height)

    end_time = time.time()
    tot_1d += end_time - start_time

end_profiler(profiler)

print(tot_1d_, tot_1d)

profiler = start_profiler()

for img1, img2 in testset:
    correlate1d_y_(img1, np_weights, output, width, height)
    correlate1d_y(img1, np_weights, weight_size, width, output)

end_profiler(profiler)

# What is all this nonsense about transposing & contiguous arrays?
# because numpy.dot is wrapping some code in {}, it prefers to run on arrays that are
# cleanly C contiguous, i.e. stored and accessed by


# compare F major vs C major

# maybe some useful numpy vs numba details (?)