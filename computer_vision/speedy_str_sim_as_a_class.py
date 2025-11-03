from .structural_sim_from_scratch import setup, generate_weights, run_math_complete, normalize_diff, correlate1d_x, correlate1d_y
from .speedy_str_sim import run_mode_rearr_y, update_corr_rearr_y
import numpy as np
import math

class Speedster:
    def __init__(self, frame_height, frame_width, sigma=1.5, truncate=3.5, data_range=255):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.ux, self.uy, self.uxx, self.uyy, self.uxy = setup(frame_height, frame_width, 'C', np.float32)
        self.ux_tmp, self.uy_tmp, self.uxx_tmp, self.uyy_tmp, self.uxy_tmp = setup(frame_height, frame_width, 'C', np.float32)
        self.out = np.zeros((frame_height, frame_width), dtype=np.uint8)

        self.update_weights(sigma, truncate)

        self.data_range = data_range

    def calc_out(self, img1, img2):
        run_mode_rearr_y(img1, self.uy_tmp, self.uy, self.uyy_tmp, self.uyy, self.size1, self.size2, self.frame_width, self.frame_height, self.np_weights, self.weight_size)
        update_corr_rearr_y(img2, img1, self.ux_tmp, self.ux, self.uxx_tmp, self.uxx, self.uxy_tmp, self.uxy, self.size1, self.size2, self.frame_width, self.frame_height,
                            self.np_weights, self.weight_size)

        bufferarr = run_math_complete(self.cov_norm, self.data_range, self.ux, self.uy, self.uxx, self.uyy, self.uxy)

        S_t = bufferarr.T

        normalize_diff(S_t, self.frame_width, self.frame_height, self.out)

    def update_weights(self, sigma, truncate):
        weights, self.cov_norm = generate_weights(ndim=2, sigma=sigma, truncate=truncate)
        weights = weights.tolist()
        self.weight_size = len(weights)
        self.size1 = math.floor(self.weight_size / 2)
        self.size2 = self.weight_size - self.size1 - 1
        self.np_weights = np.array(weights, dtype=np.float32, order='C')
