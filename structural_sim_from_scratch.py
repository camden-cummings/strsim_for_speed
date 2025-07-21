""""""
import numba as nb
import numpy as np
import scipy.ndimage as ndi

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

def setup(width, height, order):
    """Declares spaces for ux, uy, uxx, uyy & uxy.

    Parameters
    ----------
    width
        Width of image.
    height
        Height of image.
    order
        Array can be in C major or R major - depending on what
        operations are performed, one or the other may be faster.

    Returns
    -------
    unknown
    """
    ux = np.zeros((height, width), dtype=np.float32, order=order)
    uy = np.zeros((height, width), dtype=np.float32, order=order)
    uxx = np.zeros((height, width), dtype=np.float32, order=order)
    uyy = np.zeros((height, width), dtype=np.float32, order=order)
    uxy = np.zeros((height, width), dtype=np.float32, order=order)

    return ux, uy, uxx, uyy, uxy

#@nb.guvectorize([(float64, int64, float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:])], '(),(),(m,n),(m,n),(m,n),(m,n),(m,n)->(m,n)', nopython=True, target='cuda')
@nb.njit(parallel=True, fastmath=True)
def run_math_complete(cov_norm, data_range, ux, uy, uxx, uyy, uxy):
    """Use to compare images in isolation (i.e. not on a video).

    Parameters
    ----------
    cov_norm
    data_range
    ux
    uy
    uxx
    uyy
    uxy

    Returns
    -------
    unknown
    """
    ux_squared = np.multiply(ux, ux)
    uy_squared = np.multiply(uy, uy)
    ux_uy = np.multiply(ux, uy)
    sigma_x = np.multiply(cov_norm, (uxx - ux_squared))
    sigma_xy = np.multiply(cov_norm, (uxy - ux_uy))
    sigma_y = np.multiply(cov_norm, (uyy - uy_squared))

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    A1 = 2 * ux_uy + C1
    A2 = 2 * sigma_xy + C2
    B1 = ux_squared + uy_squared + C1
    B2 = sigma_x + sigma_y + C2

    return (A1 * A2) / (B1 * B2)

@nb.njit(parallel=True, fastmath=True)
def run_math_(cov_norm, data_range, ux, uy, uxx, uyy, uxy):
    """Use to compare images in isolation (i.e. not on a video).

    Parameters
    ----------
    cov_norm
    data_range
    ux
    uy
    uxx
    uyy
    uxy

    Returns
    -------
    unknown
    """
    ux_squared = np.multiply(ux, ux)
    uy_squared = np.multiply(uy, uy)
    ux_uy = np.multiply(ux, uy)
    sigma_x = np.multiply(cov_norm, (uxx - ux_squared))
    sigma_xy = np.multiply(cov_norm, (uxy - ux_uy))
    sigma_y = np.multiply(cov_norm, (uyy - uy_squared))

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    """A1 = 2 * ux_uy + C1
    A2 = 2 * sigma_xy + C2
    B1 = ux_squared + uy_squared + C1
    B2 = sigma_x + sigma_y + C2
    """

    return ((2 * ux_uy + C1) * (2 * sigma_xy + C2)) / ((ux_squared + uy_squared + C1) * (sigma_x + sigma_y + C2))

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
def normalize_diff(diff, width, height):
    """Standardize diff to range [0, 254].

    Parameters
    ----------
    diff
    width
    height

    Returns
    -------
    unknown
    """
    for x in nb.prange(height):
        for y in nb.prange(width):
            if diff[x][y] > 1:
                diff[x][y] = 1
            elif diff[x][y] < 0:
                diff[x][y] = 0

            diff[x][y] *= 255

    diff = diff.astype("uint8")
    return diff

@nb.njit(parallel=True, fastmath=True)
def __correlate1d_x(rearr, weights, weight_size, output, height):
    """Applies weights to image in x direction.

    Parameters
    ----------
    rearr
    weights
    weight_size
    output
    height

    Returns
    -------
    unknown
    """
    for start in nb.prange(height):
        end = start+weight_size
        new_arr = rearr[start:end].transpose()
        np.dot(new_arr, weights, output[start])

def correlate1d_y_wrap(matr, weights, weight_size, width, output, size1, size2):
    T = matr.transpose()
    rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)
    rearr = np.ascontiguousarray(rearr)

    __correlate1d_y(rearr, weights, weight_size, width, output)


@nb.njit(parallel=True, fastmath=True)
def __correlate1d_y(rearr, weights, weight_size, width, output):
    """Applies weights to image in y direction. For speed, the given image is expected to be transposed from

    Parameters
    ----------
    rearr
    weights
    weight_size
    width
    output

    Returns
    -------
    unknown
    """
    for start in nb.prange(width):
        end = start+weight_size
        np.dot(weights, rearr[start:end], output[start])
