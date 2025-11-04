""""""
import numba as nb
import numpy as np
import scipy.ndimage as ndi
import scipy

def generate_weights(ndim=2, sigma=1.5, truncate=3.5):
    """Generates Gaussian weights based on given sigma and truncate.

    Parameters
    ----------
    ndim
        Dimension of image.
    sigma
    truncate

    Returns
    -------
    unknown
    """
    radius = int(truncate * sigma + 0.5)  # radius as in ndimage
    weights = ndi._filters._gaussian_kernel1d(sigma, 0, radius)[::-1]

    window_size = 2 * radius + 1
    NP = window_size**ndim
    cov_norm = NP / (NP - 1)  # sample covariance

    return weights, cov_norm

def setup(width, height, order, data_type):
    """Declares spaces for ux, uy, uxx, uyy & uxy.

    Parameters
    ----------
    width
        Width of image.
    height
        Height of image.
    order
        Array can be in C major or F major - depending on what
        operations are performed, one or the other may be faster.

    Returns
    -------
    unknown
    """
    ux = np.zeros((height, width), dtype=data_type, order=order)
    uy = np.zeros((height, width), dtype=data_type, order=order)
    uxx = np.zeros((height, width), dtype=data_type, order=order)
    uyy = np.zeros((height, width), dtype=data_type, order=order)
    uxy = np.zeros((height, width), dtype=data_type, order=order)

    return ux, uy, uxx, uyy, uxy

@nb.njit(parallel=True, fastmath=True)
def run_math_complete(cov_norm, data_range, ux, uy, uxx, uyy, uxy):
    """
    Use to compare images in isolation (i.e. not on a video).
    """

    ux_squared = np.multiply(ux, ux)
    uy_squared = np.multiply(uy, uy)
    ux_uy = np.multiply(ux, uy)

    sigma_x = np.multiply(cov_norm, (uxx - ux_squared))
    sigma_xy = np.multiply(cov_norm, (uxy - ux_uy))
    sigma_y = np.multiply(cov_norm, (uyy - uy_squared))

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

#    print("rm", np.min(2 * ux_uy + C1), np.max(2 * ux_uy + C1))
    A1 = 2 * ux_uy + C1
    A2 = 2 * sigma_xy + C2
    B1 = ux_squared + uy_squared + C1
    B2 = sigma_x + sigma_y + C2

    return (A1 * A2) / (B1 * B2)

@nb.njit(parallel=True, fastmath=True)
def run_math(cov_norm, data_range, ux, uy, uxx, sigma_y, uxy):
    """If you are running tracking on a video, and you are comparing each image to the prior image - you can keep track of
    vy s.t. you don't have to recalculate it each time, because your old uy will become your new ux.

    Parameters
    ----------
    cov_norm
    data_range
    ux
    uy
    uxx
    sigma_y
    uxy

    Returns
    -------
    unknown
    """
    ux_squared = np.multiply(ux, ux)
    uy_squared = np.multiply(uy, uy)
    ux_uy = np.multiply(ux, uy)
    sigma_x = np.multiply(cov_norm, (uxx-ux_squared))
    sigma_xy = np.multiply(cov_norm, (uxy - ux_uy))

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    A1 = np.multiply(ux_uy,2) + C1
    A2 = np.multiply(sigma_xy,2) + C2
    B1 = ux_squared + uy_squared + C1
    B2 = sigma_x + sigma_y + C2

    return (A1 * A2) / (B1 * B2)

@nb.njit(parallel=True, fastmath=True)
def normalize_diff(diff, width, height, out):
    for x in nb.prange(height):
        for y in nb.prange(width):
            if diff[x][y] > 1:
                out[x][y] = 255
            elif diff[x][y] < 0:
                out[x][y] = 0
            else:
                out[x][y] = diff[x][y] * 255

    out = out.astype("uint8")

def correlate1d_x(inp, size1, size2, np_weights, weight_size, height, out):
    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
    nb_correlate1d_x(rearr, np_weights, weight_size, height, out)

def correlate1d_y(inp, size1, size2, np_weights, weight_size, width, out):
    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
    nb_correlate1d_y(rearr, np_weights, weight_size, width, out)

@nb.njit(parallel=True, fastmath=True)
def nb_correlate1d_x(rearr, weights, weight_size, height, output):
    for start in nb.prange(height):
        end = start+weight_size
        output[:, start] = np.dot(weights, rearr[start:end])

@nb.njit(parallel=True, fastmath=True)
def nb_correlate1d_y(rearr, weights, weight_size, width, output):
    """Applies weights to image in y direction. For speed, the given image is expected to be transposed from

    Parameters
    ----------
    rearr
    weights
    weight_size
    width
    output

    """

    for start in nb.prange(width):
        end = start+weight_size
        np.dot(weights, rearr[start:end], output[start])