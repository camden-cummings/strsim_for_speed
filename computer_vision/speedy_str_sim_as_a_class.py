from .structural_sim_from_scratch import setup, generate_weights, run_math_complete, normalize_diff, nb_correlate1d_x, nb_correlate1d_y, feed_rearr
import numpy as np
import math

class SpeedyCV:
    def __init__(self, frame_height, frame_width, sigma=1.5, truncate=3.5, data_range=255):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.ux, self.uy, self.uxx, self.uyy, self.uxy = setup(frame_height, frame_width, 'C', np.float32)
        self.ux_tmp, self.uy_tmp, self.uxx_tmp, self.uyy_tmp, self.uxy_tmp = setup(frame_height, frame_width, 'C', np.float32)
        self.out = np.zeros((frame_height, frame_width), dtype=np.uint8)

        self.update_weights(sigma, truncate)
        
        self.tmp = np.zeros((self.frame_width, self.frame_height), dtype=np.float32)

        self.data_range = data_range

    def calc_out(self, img1, img2):
        self.run_corr(img1)
        self.run_mode(img2)
        self.run_against(img1, img2)

        bufferarr = run_math_complete(self.cov_norm, self.data_range, self.ux, self.uy, self.uxx, self.uyy, self.uxy)

        S_t = bufferarr.T

        normalize_diff(S_t, self.frame_width, self.frame_height, self.out)

    def update_weights(self, sigma, truncate):
        radius = int(truncate * sigma + 0.5)
        if radius > 1:
            weights, self.cov_norm = generate_weights(ndim=2, sigma=sigma, truncate=truncate)
            weights = weights.tolist()
            self.weight_size = len(weights)
            self.size1 = math.floor(self.weight_size / 2)
            self.size2 = self.weight_size - self.size1 - 1
            self.np_weights = np.array(weights, dtype=np.float32, order='C')
            
            self.tmp_x = np.zeros((self.frame_height+self.size1+self.size2, self.frame_width), dtype=np.float32)
            self.tmp_y = np.zeros((self.frame_width+self.size1+self.size2, self.frame_height), dtype=np.float32)
        else:
            print('WARNING: radius is one, cannot generate weights using this sigma and truncate. Keeping previous weights.')
     
    """
    def correlate1d_x(self, inp, out):
        rearr = np.concatenate((inp[0:self.size1][::-1], inp, inp[-self.size2:][::-1]))
        nb_correlate1d_x(rearr, self.np_weights, self.weight_size, self.frame_height, out)
    
    def correlate1d_y(self, inp, out):
        rearr = np.concatenate((inp[0:self.size1][::-1], inp, inp[-self.size2:][::-1]))
        nb_correlate1d_y(rearr, self.np_weights, self.weight_size, self.frame_width, out)
    """
    
    def correlate1d_x(self, inp, out, rearr):
        feed_rearr(inp, rearr, self.frame_height, self.size1, self.size2)
        nb_correlate1d_x(rearr, self.np_weights, self.weight_size, self.frame_height, out)
    
    def correlate1d_y(self, inp, out, rearr):
        feed_rearr(inp, rearr, self.frame_width, self.size1, self.size2)
        nb_correlate1d_y(rearr, self.np_weights, self.weight_size, self.frame_width, out)

    def run_corr(self, curr_img):  
        self.correlate1d_x(curr_img, self.ux_tmp, self.tmp_x)
        self.correlate1d_y(self.ux_tmp, self.ux, self.tmp_y)
        
        inp = np.multiply(curr_img, curr_img)
        self.correlate1d_x(inp, self.uxx_tmp, self.tmp_x)
        self.correlate1d_y(self.uxx_tmp, self.uxx, self.tmp_y)
    
    def run_mode(self, mode_img):        
        self.correlate1d_x(mode_img, self.uy_tmp, self.tmp_x)
        self.correlate1d_y(self.uy_tmp, self.uy, self.tmp_y)
    
        inp = np.multiply(mode_img, mode_img)
        self.correlate1d_x(inp, self.uyy_tmp, self.tmp_x)
        self.correlate1d_y(self.uyy_tmp, self.uyy, self.tmp_y)
        
    def run_against(self, img1, img2):
        inp = np.multiply(img1, img2)
        self.correlate1d_x(inp, self.uxy_tmp, self.tmp_x)
        self.correlate1d_y(self.uxy_tmp, self.uxy, self.tmp_y)

    def calc_out_from_existing_arrays(self):
        bufferarr = run_math_complete(self.cov_norm, self.data_range, self.ux, self.uy, self.uxx, self.uyy, self.uxy)
        S_t = bufferarr.T
        normalize_diff(S_t, self.frame_width, self.frame_height, self.out)