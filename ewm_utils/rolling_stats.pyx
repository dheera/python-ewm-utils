# rolling_stats.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
cimport numpy as np
from libc.math cimport fabs, isnan
import math

cdef class RollingStats:
    """
    Compute rolling exponentially weighted statistics (mean and std).
    """
    cdef public double alpha
    cdef public bint adjust
    cdef public int min_periods

    # For explicit mode:
    cdef list values

    # For recursive mode:
    cdef public int count
    cdef double mean    # current mean
    cdef double v       # current uncorrected variance

    def __init__(self, double alpha, bint adjust=True, int min_periods=1):
        if not (alpha > 0 and alpha <= 1):
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.adjust = adjust
        self.min_periods = min_periods
        if self.adjust:
            self.values = []
        else:
            self.count = 0
            self.mean = 0.0
            self.v = 0.0

    cpdef tuple update(self, double value):
        cdef int n, i
        cdef np.ndarray weights, arr
        cdef double sum_w, mean_val, uncorrected_var, correction, std_val
        cdef double old_mean  # declared at the start for recursive mode
        if self.adjust:
            self.values.append(value)
            n = len(self.values)
            if n < self.min_periods:
                return (np.nan, np.nan)
            weights = np.empty(n, dtype=np.float64)
            for i in range(n):
                weights[i] = (1 - self.alpha) ** (n - 1 - i)
            sum_w = weights.sum()
            arr = np.array(self.values, dtype=np.float64)
            mean_val = (arr * weights).sum() / sum_w
            uncorrected_var = ((arr - mean_val)**2 * weights).sum() / sum_w
            correction = 1 - ((weights**2).sum() / (sum_w**2))
            if correction <= 0:
                std_val = np.nan
            else:
                std_val = math.sqrt(uncorrected_var / correction)
            return (mean_val, std_val)
        else:
            self.count += 1
            if self.count == 1:
                self.mean = value
                self.v = 0.0
                return (np.nan, np.nan)
            else:
                old_mean = self.mean
                self.mean = old_mean + self.alpha * (value - old_mean)
                self.v = (1 - self.alpha) * (self.v + self.alpha * (value - old_mean)**2)
                if self.count < self.min_periods:
                    return (np.nan, np.nan)
                return (self.mean, math.sqrt(self.v))

