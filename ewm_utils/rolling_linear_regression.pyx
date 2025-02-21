# ewm_rolling_linear_regression.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
cimport numpy as np
from libc.math cimport fabs, isnan, INFINITY
import random

ctypedef np.double_t DTYPE_t

###############################################################################
#              Helper functions for weighted regression                     #
###############################################################################

cdef tuple _weighted_linear_regression(np.ndarray[DTYPE_t, ndim=1] x,
                                        np.ndarray[DTYPE_t, ndim=1] y,
                                        np.ndarray[DTYPE_t, ndim=1] weights):
    """
    Compute weighted least squares estimates for slope and intercept.
    Returns (slope, intercept).
    """
    cdef int n = x.shape[0]
    cdef double sum_w = 0.0, mean_x = 0.0, mean_y = 0.0, cov = 0.0, var_x = 0.0
    cdef int i
    for i in range(n):
        sum_w += weights[i]
    if sum_w == 0:
        return (np.nan, np.nan)
    for i in range(n):
        mean_x += x[i] * weights[i]
        mean_y += y[i] * weights[i]
    mean_x /= sum_w
    mean_y /= sum_w
    for i in range(n):
        cov += weights[i] * (x[i] - mean_x) * (y[i] - mean_y)
        var_x += weights[i] * (x[i] - mean_x) * (x[i] - mean_x)
    if var_x == 0:
        return (np.nan, np.nan)
    return (cov / var_x, mean_y - (cov / var_x) * mean_x)

cdef double _weighted_r2(np.ndarray[DTYPE_t, ndim=1] x,
                         np.ndarray[DTYPE_t, ndim=1] y,
                         double slope, double intercept,
                         np.ndarray[DTYPE_t, ndim=1] weights):
    """
    Compute the weighted R² given the model (slope, intercept).
    """
    cdef int n = x.shape[0]
    cdef double sum_w = 0.0, mean_y = 0.0, TSS = 0.0, RSS = 0.0, r2
    cdef int i
    for i in range(n):
        sum_w += weights[i]
    if sum_w == 0:
        return np.nan
    for i in range(n):
        mean_y += y[i] * weights[i]
    mean_y /= sum_w
    for i in range(n):
        TSS += weights[i] * (y[i] - mean_y) * (y[i] - mean_y)
        RSS += weights[i] * (y[i] - (slope * x[i] + intercept))**2
    if TSS > 0:
        r2 = 1 - RSS / TSS
    else:
        r2 = np.nan
    return r2

cdef tuple _ransac_regression(np.ndarray[DTYPE_t, ndim=1] x,
                                np.ndarray[DTYPE_t, ndim=1] y,
                                np.ndarray[DTYPE_t, ndim=1] weights,
                                int ransac_iterations,
                                double ransac_threshold,
                                int min_periods):
    """
    Perform a simple RANSAC procedure for robust regression.
    Randomly selects two distinct points, computes a candidate line,
    and evaluates inliers based on residuals.
    Returns (slope, intercept) of the best candidate, or (np.nan, np.nan).
    """
    cdef int n = x.shape[0]
    if n < 2:
        return (np.nan, np.nan)
    cdef double best_error = INFINITY
    cdef double candidate_slope, candidate_intercept
    cdef double y_pred, res, median_resid, thresh, inlier_error
    cdef int i, j, inlier_count
    cdef np.ndarray residuals = np.empty(n, dtype=np.float64)
    cdef np.ndarray best_mask = None
    cdef np.ndarray current_mask
    cdef int idx0, idx1, idx2

    for i in range(ransac_iterations):
        idx0 = random.randrange(n)
        idx1 = random.randrange(n)
        while idx1 == idx0:
            idx1 = random.randrange(n)
        if x[idx1] == x[idx0]:
            continue
        idx2 = random.randrange(n)
        while idx2 == idx0 or idx2 == idx1:
            idx2 = random.randrange(n)
        candidate_slope = (y[idx1] - y[idx0]) / (x[idx1] - x[idx0])
        candidate_intercept = y[idx0] - candidate_slope * x[idx0]
        for j in range(n):
            y_pred = candidate_slope * x[j] + candidate_intercept
            res = fabs(y[j] - y_pred)
            residuals[j] = res
        median_resid = np.median(residuals)
        thresh = ransac_threshold * (median_resid if median_resid > 0 else 1.0)
        current_mask = (residuals < thresh)
        inlier_count = np.sum(current_mask)
        if inlier_count < max(2, min_periods):
            continue
        inlier_error = np.sum(weights[current_mask] * residuals[current_mask])
        if inlier_error < best_error:
            best_error = inlier_error
            best_mask = current_mask.copy()
    if best_mask is not None and np.sum(best_mask) >= max(2, min_periods):
        return _weighted_linear_regression(x[best_mask], y[best_mask], weights[best_mask])
    else:
        return (np.nan, np.nan)

###############################################################################
#                 Rolling  Linear Regression Class                       #
###############################################################################

cdef class RollingLinearRegression:
    cdef public double alpha
    cdef public bint adjust
    cdef public int min_periods
    cdef public object outlier_method  # str or None
    cdef public int ransac_iterations
    cdef public double ransac_threshold

    # For explicit mode: store all data points.
    cdef list xs
    cdef list ys

    # For recursive mode:
    cdef public int count
    cdef double S0, S1, S2, Sy, Sxy, Syy

    def __init__(self, double alpha, bint adjust=True, int min_periods=1,
                 outlier_method=None, int ransac_iterations=100, double ransac_threshold=2.0):
        if not (alpha > 0 and alpha <= 1):
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.adjust = adjust
        self.min_periods = min_periods
        self.outlier_method = outlier_method
        self.ransac_iterations = ransac_iterations
        self.ransac_threshold = ransac_threshold

        if self.adjust:
            self.xs = []
            self.ys = []
        else:
            self.count = 0
            self.S0 = 0.0
            self.S1 = 0.0
            self.S2 = 0.0
            self.Sy = 0.0
            self.Sxy = 0.0
            self.Syy = 0.0

    cpdef tuple update(self, double x, double y):
        """
        Update the regression with a new (x, y) data point.
        Returns (slope, intercept, r2).
        If there are fewer than min_periods data points, returns NaNs.
        """
        # Declare local variables at the start.
        cdef np.ndarray weights, x_arr, y_arr
        cdef int n, i
        cdef double slope, intercept, r2, mean_x, mean_y, var_x, cov
        cdef double TSS, RSS

        if self.adjust:
            # Explicit mode: append new data.
            self.xs.append(x)
            self.ys.append(y)
            n = len(self.xs)
            if n < self.min_periods:
                return (np.nan, np.nan, np.nan)
            weights = np.empty(n, dtype=np.float64)
            for i in range(n):
                weights[i] = (1 - self.alpha) ** (n - 1 - i)
            x_arr = np.array(self.xs, dtype=np.float64)
            y_arr = np.array(self.ys, dtype=np.float64)
            if self.outlier_method == 'ransac':
                slope, intercept = _ransac_regression(x_arr, y_arr, weights,
                                                       self.ransac_iterations,
                                                       self.ransac_threshold,
                                                       self.min_periods)
            else:
                slope, intercept = _weighted_linear_regression(x_arr, y_arr, weights)
            r2 = _weighted_r2(x_arr, y_arr, slope, intercept, weights)
            return (slope, intercept, r2)
        else:
            # Recursive mode.
            self.count += 1
            if self.count == 1:
                self.S0 = 1.0
                self.S1 = x
                self.S2 = x**2
                self.Sy = y
                self.Sxy = x*y
                self.Syy = y**2
                return (np.nan, np.nan, np.nan)
            else:
                self.S0 = (1 - self.alpha) * self.S0 + 1.0
                self.S1 = (1 - self.alpha) * self.S1 + x
                self.S2 = (1 - self.alpha) * self.S2 + x**2
                self.Sy = (1 - self.alpha) * self.Sy + y
                self.Sxy = (1 - self.alpha) * self.Sxy + x*y
                self.Syy = (1 - self.alpha) * self.Syy + y**2
                if self.count < self.min_periods:
                    return (np.nan, np.nan, np.nan)
                mean_x = self.S1 / self.S0
                mean_y = self.Sy / self.S0
                var_x = self.S2 / self.S0 - mean_x**2
                cov = self.Sxy / self.S0 - mean_x * mean_y
                if var_x > 0:
                    slope = cov / var_x
                    intercept = mean_y - slope * mean_x
                else:
                    slope = np.nan
                    intercept = np.nan
                TSS = self.Syy / self.S0 - mean_y**2
                RSS = (self.Syy / self.S0) - 2*(slope*(self.Sxy/self.S0) + intercept*(self.Sy/self.S0)) \
                      + (slope**2*(self.S2/self.S0) + 2*slope*intercept*(self.S1/self.S0) + intercept**2)
                if TSS > 0:
                    r2 = 1 - RSS / TSS
                else:
                    r2 = np.nan
                return (slope, intercept, r2)

