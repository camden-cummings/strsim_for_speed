import numba as nb
import numpy as np
import math

def structural_similarity(
        im1,
        im2,
        data_range,
        weights):
    ndim = im1.ndim

    # ndimage filters need floating point data

    im1 = im1.astype(np.float64, copy=False)
    im2 = im2.astype(np.float64, copy=False)

    sigma = 1.5
    truncate = 3.5
    r = int(truncate * sigma + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    NP = win_size**ndim
    cov_norm = NP / (NP - 1)  # sample covariance

    ux = np.zeros((1200, 1760))
    uy = np.zeros((1200, 1760))
    uxx = np.zeros((1200, 1760))
    uyy = np.zeros((1200, 1760))
    uxy = np.zeros((1200, 1760))

    correlate1d(im1, weights, ux)
    correlate1d(im2, weights, uy)

    correlate1d(im1 * im1, weights, uxx)
    correlate1d(im2 * im2, weights, uyy)
    correlate1d(im1 * im2, weights, uxy)

    return run_math(cov_norm, data_range, ux, uy, uxx, uyy, uxy)

@nb.njit(parallel=True, fastmath=True)
def run_math(cov_norm, data_range, ux, uy, uxx, uyy, uxy):
    K1 = 0.01
    K2 = 0.03

    ux_squared = ux * ux
    uy_squared = uy * uy
    ux_uy = ux * uy
    vx = cov_norm * (uxx - ux_squared)
    vy = cov_norm * (uyy - uy_squared)
    vxy = cov_norm * (uxy - ux_uy)

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2


    A1, A2, B1, B2 = (
        2 * ux_uy + C1,
        2 * vxy + C2,
        ux_squared + uy_squared + C1,
        vx + vy + C2,
    )

    return (A1 * A2) / (B1 * B2)

@nb.njit(parallel=True, fastmath=True)
def correlate1d(input, weights, output=None):
    height, width = (1200, 1760)
    weight_size = len(weights)
    size1 = math.floor(weight_size / 2)
    size2 = weight_size - size1 - 1

    """
    symmetric = 0
    if weight_size % 2 == 1:  # if the input weight array is even, it will be symmetric = 0, so we don't need to run this calculation
        if all(weights == weights[::-1]):
            symmetric = 1
        elif all(weights == -weights[::-1]):  # i believe this is ok but don't trust it 100%
            symmetric = -1
    """
    symmetric = 1

    for ii in nb.prange(height):
        np_row = input[ii]

        size1_arr = np_row[0:size1][::-1]
        size2_arr = np_row[-size2:][::-1]

        new_arr = np.concatenate((size1_arr, np_row, size2_arr))

        if symmetric > 0:
            for start in range(width):
                n = start+size1
                total_neighbour = weights[size1]*new_arr[n]
                for x in range(1, size1+1):
                    total_neighbour += (new_arr[n+x] + new_arr[n-x]) * weights[size1+x]

                output[ii][start] = total_neighbour

        elif symmetric < 0:
            pass
        else:
            for start in range(len(new_arr) - size1 - 1):
                output[ii][start] = new_arr[start + size1 + 1] * weights[-1]
                for i in range(start, start + size1 + 1):
                    output[ii][start] += new_arr[i] * weights[i - start]