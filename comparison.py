from computer_vision.profile_wrap import profilefunc
from computer_vision.speedy_str_sim_as_a_class import SpeedyCV
from computer_vision.structural_sim_from_scratch import correlate1d_x, correlate1d_y

from skimage.metrics import structural_similarity as ssim
import numpy as np

width = 992
height = 660
SAMPLE_SIZE = 1000

spd = SpeedyCV(height, width)

img1 = np.array(np.random.randint(100, size=(height, width)), dtype=np.float32)
img2 = np.array(np.random.randint(100, size=(height, width)), dtype=np.float32)

io = np.array(np.random.randint(100, size=(width, height)), dtype=np.float32)

output = np.zeros((width, height), dtype=np.float32)

rearr1 = np.zeros((height+10, width), dtype=np.float32)
rearr2 = np.zeros((width+10, height), dtype=np.float32)

### compiling ###

spd.calc_out(img1, img2)

spd.correlate1d_x(img1, output)
spd.correlate1d_y(io, output)
spd.correlate1d_x_(img1, output, rearr1)
spd.correlate1d_y_(io, output, rearr2)

### compiling ###

#correlate 1d x comparison
profilefunc(spd.correlate1d_x, SAMPLE_SIZE, img1, output)
profilefunc(spd.correlate1d_x_, SAMPLE_SIZE, img1, output, rearr1)
#profilefunc(correlate1d, SAMPLE_SIZE, img1, weights, axis=0, output=output1, mode='reflect', cval=0.0, origin=0)

#correlate 1d y comparison
profilefunc(spd.correlate1d_y, SAMPLE_SIZE, io, output)
profilefunc(spd.correlate1d_y_, SAMPLE_SIZE, io, output, rearr2)
#profilefunc(correlate1d, SAMPLE_SIZE, T, weights, axis=0, output=output2, mode='reflect', cval=0.0, origin=0)

#structural similarity comparison
profilefunc(spd.calc_out, SAMPLE_SIZE, img1, img2)
profilefunc(spd.calc_out_, SAMPLE_SIZE, img1, img2)

profilefunc(ssim, SAMPLE_SIZE, img1, img2, data_range=255, full=True)
