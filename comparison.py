from computer_vision.profile_wrap import profilefunc
from computer_vision.speedy_str_sim_as_a_class import SpeedyCV

from skimage.metrics import structural_similarity as ssim

import numpy as np

width = 992
height = 660
SAMPLE_SIZE = 1000

spd = SpeedyCV(height, width)

img1 = np.array(np.random.randint(100, size=(height, width)), dtype=np.float32)
img2 = np.array(np.random.randint(100, size=(height, width)), dtype=np.float32)

output = np.zeros((width, height), dtype=np.float32)

### compiling ###

spd.calc_out(img1, img2)
ssim(img1, img2, data_range=255, full=True)

### compiling ###

#structural similarity comparison
profilefunc(spd.calc_out, SAMPLE_SIZE, img1, img2)
profilefunc(ssim, SAMPLE_SIZE, img1, img2, data_range=255, full=True)
