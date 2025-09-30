from computer_vision.profile_wrap import profilefunc
from computer_vision.structural_sim_from_scratch import setup, generate_weights, correlate1d_x, correlate1d_y, normalize_diff
from computer_vision.speedy_str_sim import run_correlate_rearr_y

from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import correlate1d

import math

import numpy as np

width = 1760
height = 1200
SAMPLE_SIZE = 100

ux_tmp, uy_tmp, uxx_tmp, uyy_tmp, uxy_tmp = setup(width, height, 'C')
ux, uy, uxx, uyy, uxy = setup(height, width,'C')

weights, cov_norm = generate_weights(ndim=2, sigma=1.5, truncate=3.5)
np_weights = np.array(weights, dtype=np.float32, order='C')

weight_size = len(weights)
size1 = math.floor(weight_size / 2)
size2 = weight_size - size1 - 1

img1 = np.array(np.random.randint(100, size=(height, width)), dtype=np.float32)

#img_path = "/home/chamomile/Thyme-lab/data/combined_vids_csvs/10dpf/2024-02-16_1run_2024-02-16-112634-0000.avi-mode.png"
#img1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
#img1 = img1.astype(np.float32, copy=False, order='C')

rearr1 = np.concatenate((img1[0:size1][::-1], img1, img1[-size2:][::-1]))

img2 = np.array(np.random.randint(100, size=(height, width)), dtype=np.float32)
T = img2.T
rearr2 = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)

output1 = np.zeros((height, width), dtype=np.float32)
output2 = np.zeros((width, height), dtype=np.float32)

### compiling ###

correlate1d_x(rearr1, np_weights, weight_size, height, output1)
correlate1d_y(rearr2, np_weights, weight_size, width, output2)

run_correlate_rearr_y(img1, img2, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, uy_tmp,
                      uy, uyy_tmp, uyy, size1,
                      size2, width, height, cov_norm, 255, np_weights, weight_size, output1)

### compiling ###

#correlate 1d x comparison
profilefunc(correlate1d_x, SAMPLE_SIZE,  rearr1, np_weights, weight_size, height, output1)
profilefunc(correlate1d, SAMPLE_SIZE, img1, weights, axis=0, output=output1, mode='reflect', cval=0.0, origin=0)

#correlate 1d y comparison
profilefunc(correlate1d_y, SAMPLE_SIZE, rearr2, np_weights, weight_size, width, output2)
profilefunc(correlate1d, SAMPLE_SIZE, T, weights, axis=0, output=output2, mode='reflect', cval=0.0, origin=0)

#structural similarity comparison
profilefunc(run_correlate_rearr_y, SAMPLE_SIZE, img1, img2, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, uy_tmp,
                      uy, uyy_tmp, uyy, size1,
                      size2, width, height, cov_norm, 255, np_weights, weight_size, output1)
profilefunc(ssim, SAMPLE_SIZE, img1, img2, data_range=255, full=True)
