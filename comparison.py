"""Alternative versions of correlate1d."""
import numba as nb
from numba import float64
import numpy as np
import math

@nb.njit(parallel=True, fastmath=True)
def correlate1d_x(input, weights, output, width, height):
    weight_size = len(weights)
    size1 = math.floor(weight_size / 2)
    size2 = weight_size - size1 - 1

    # rearr = np.concatenate((input[0:size1][::-1], input, input[-size2:][::-1]))
    for jj in nb.prange(width):
        np_row = input[:, jj]
        size1_arr = np_row[0:size1][::-1]
        size2_arr = np_row[-size2:][::-1]
        new_arr = np.concatenate((size1_arr, np_row, size2_arr))
        # new_arr = rearr[:, jj]

        for start in range(height):
            n = start + size1
            total_neighbour = weights[size1] * new_arr[n]
            for x in range(1, size1 + 1):
                total_neighbour += (new_arr[n + x] + new_arr[n - x]) * weights[size1 + x]
            output[start][jj] = total_neighbour

#@nb.guvectorize([(float64[:,:], float64[:], float64, float64, float64[:, :])], "(m,n), (o), (), () -> (m,n)", fastmath=True, nopython=True)

@nb.njit(parallel=True, fastmath=True)
def correlate1d_x_vector(input, weights, width, height, output):
    weight_size = len(weights)
    size1 = math.floor(weight_size / 2)
    size2 = weight_size - size1 - 1

    # rearr = np.concatenate((input[0:size1][::-1], input, input[-size2:][::-1]))
    for jj in nb.prange(width):
        np_row = input[:, jj]
        size1_arr = np_row[0:size1][::-1]
        size2_arr = np_row[-size2:][::-1]
        new_arr = np.concatenate((size1_arr, np_row, size2_arr))
        # new_arr = rearr[:, jj]

        for start in range(height):
            n = start + size1
            total_neighbour = weights[size1] * new_arr[n]
            for x in range(1, size1 + 1):
                total_neighbour += (new_arr[n + x] + new_arr[n - x]) * weights[size1 + x]
            output[start][jj] = total_neighbour

@nb.njit(parallel=True, fastmath=True)
def correlate1d_y(input, weights, output, width, height):
    weight_size = len(weights)
    size1 = math.floor(weight_size / 2)
    size2 = weight_size - size1 - 1

    # rearr = np.concatenate((input[:, 0:size1][::-1], input, input[:, -size2:][::-1]), axis=1) # something in the construction of this is wrong but I can't figure out what
    for ii in nb.prange(height):
        np_row = input[ii]
        size1_arr = np_row[0:size1][::-1]
        size2_arr = np_row[-size2:][::-1]
        new_arr = np.concatenate((size1_arr, np_row, size2_arr))

        for start in range(width):
            n = start + size1
            total_neighbour = weights[size1] * new_arr[n]
            for x in range(1, size1 + 1):
                total_neighbour += (new_arr[n + x] + new_arr[n - x]) * weights[size1 + x]
            output[ii][start] = total_neighbour

@nb.guvectorize([(float64[:,:], float64[:], float64, float64, float64[:, :])], "(m,n), (o), (), () -> (m,n)", fastmath=True, nopython=True)
def correlate1d_y_vector(input, weights, width, height, output):
    weight_size = len(weights)
    size1 = math.floor(weight_size / 2)
    size2 = weight_size - size1 - 1

    # rearr = np.concatenate((input[:, 0:size1][::-1], input, input[:, -size2:][::-1]), axis=1) # something in the construction of this is wrong but I can't figure out what
    for ii in nb.prange(height):
        np_row = input[ii]
        size1_arr = np_row[0:size1][::-1]
        size2_arr = np_row[-size2:][::-1]
        new_arr = np.concatenate((size1_arr, np_row, size2_arr))

        for start in range(width):
            n = start + size1
            total_neighbour = weights[size1] * new_arr[n]
            for x in range(1, size1 + 1):
                total_neighbour += (new_arr[n + x] + new_arr[n - x]) * weights[size1 + x]
            output[ii][start] = total_neighbour