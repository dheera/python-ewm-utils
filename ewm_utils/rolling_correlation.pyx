# rolling_corr.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
cimport numpy as np
from libc.math cimport sqrt
import math
import random

ctypedef np.double_t DTYPE_t

cdef class RollingCorrelation:
    """
    Rolling exponentially weighted correlation.
    """
    cdef public double alpha
    cdef public bint adjust
    cdef public int min_periods

    # Explicit mode: store all data points.
    cdef list xs
    cdef list ys

    # Recursive mode: maintain running state.
    cdef public int count
    cdef double mean_x, mean_y, var_x, var_y, cov

    def __init__(self, double alpha, bint adjust=True, int min_periods=1):
        if not (alpha > 0 and alpha <= 1):
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.adjust = adjust
        self.min_periods = min_periods
        if self.adjust:
            self.xs = []
            self.ys = []
        else:
            self.count = 0
            self.mean_x = 0.0
            self.mean_y = 0.0
            self.var_x = 0.0
            self.var_y = 0.0
            self.cov = 0.0

    cpdef double update(self, double x, double y):
        # Declare all local C variables at the top.
        cdef int n, i
        cdef np.ndarray weights = None, x_arr = None, y_arr = None
        cdef double sum_w = 0.0, mean_x_val = 0.0, mean_y_val = 0.0
        cdef double cov_val = 0.0, var_x_val = 0.0, var_y_val = 0.0, corr = np.nan
        cdef double delta_x, delta_y

        if self.adjust:
            # Explicit mode: store new data point.
            self.xs.append(x)
            self.ys.append(y)
            n = len(self.xs)
            if n < self.min_periods:
                return np.nan
            weights = np.empty(n, dtype=np.float64)
            for i in range(n):
                weights[i] = (1 - self.alpha) ** (n - 1 - i)
            sum_w = weights.sum()
            x_arr = np.array(self.xs, dtype=np.float64)
            y_arr = np.array(self.ys, dtype=np.float64)
            mean_x_val = (x_arr * weights).sum() / sum_w
            mean_y_val = (y_arr * weights).sum() / sum_w
            cov_val = (weights * (x_arr - mean_x_val) * (y_arr - mean_y_val)).sum() / sum_w
            var_x_val = (weights * (x_arr - mean_x_val)**2).sum() / sum_w
            var_y_val = (weights * (y_arr - mean_y_val)**2).sum() / sum_w
            if var_x_val > 0 and var_y_val > 0:
                corr = cov_val / math.sqrt(var_x_val * var_y_val)
            else:
                corr = np.nan
            return corr
        else:
            # Recursive mode.
            self.count += 1
            if self.count == 1:
                self.mean_x = x
                self.mean_y = y
                self.var_x = 0.0
                self.var_y = 0.0
                self.cov = 0.0
                return np.nan
            else:
                delta_x = x - self.mean_x
                delta_y = y - self.mean_y
                self.mean_x = self.mean_x + self.alpha * delta_x
                self.mean_y = self.mean_y + self.alpha * delta_y
                self.cov = (1 - self.alpha) * (self.cov + self.alpha * delta_x * delta_y)
                self.var_x = (1 - self.alpha) * (self.var_x + self.alpha * delta_x**2)
                self.var_y = (1 - self.alpha) * (self.var_y + self.alpha * delta_y**2)
                if self.count < self.min_periods or self.var_x <= 0 or self.var_y <= 0:
                    return np.nan
                corr = self.cov / math.sqrt(self.var_x * self.var_y)
                return corr

