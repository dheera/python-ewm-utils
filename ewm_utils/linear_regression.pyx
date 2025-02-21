# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
cimport numpy as np
from libc.math cimport fabs, isnan, INFINITY
import random

ctypedef np.double_t DTYPE_t

# Helper function to compute the median of a 1D NumPy array.
cdef double _median(np.ndarray[DTYPE_t, ndim=1] arr):
    cdef int n = arr.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] b = arr.copy()  # make a copy to sort
    cdef int i, j
    cdef double key
    # Simple insertion sort
    for i in range(1, n):
        key = b[i]
        j = i - 1
        while j >= 0 and b[j] > key:
            b[j+1] = b[j]
            j -= 1
        b[j+1] = key
    if n % 2 == 1:
        return b[n//2]
    else:
        return 0.5 * (b[n//2 - 1] + b[n//2])

# Standard weighted linear regression.
cpdef tuple weighted_linear_regression(np.ndarray[DTYPE_t, ndim=1] x,
                                         np.ndarray[DTYPE_t, ndim=1] y,
                                         np.ndarray[DTYPE_t, ndim=1] weights):
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
    return (cov/var_x, mean_y - (cov/var_x)*mean_x)

# Compute weighted R^2.
cpdef double weighted_r2(np.ndarray[DTYPE_t, ndim=1] x,
                         np.ndarray[DTYPE_t, ndim=1] y,
                         double slope, double intercept,
                         np.ndarray[DTYPE_t, ndim=1] weights):
    cdef int n = x.shape[0]
    cdef double sum_w = 0.0, y_mean = 0.0, TSS = 0.0, RSS = 0.0
    cdef int i
    for i in range(n):
        sum_w += weights[i]
    if sum_w == 0:
        return np.nan
    for i in range(n):
        y_mean += y[i] * weights[i]
    y_mean /= sum_w
    for i in range(n):
        TSS += weights[i] * (y[i] - y_mean) * (y[i] - y_mean)
        RSS += weights[i] * (y[i] - (slope * x[i] + intercept)) * (y[i] - (slope * x[i] + intercept))
    if TSS > 0:
        return 1 - RSS/TSS
    else:
        return np.nan

# Robust regression using a RANSAC procedure.
cpdef tuple ransac_regression(np.ndarray[DTYPE_t, ndim=1] x,
                              np.ndarray[DTYPE_t, ndim=1] y,
                              np.ndarray[DTYPE_t, ndim=1] weights,
                              int ransac_iterations,
                              double ransac_threshold,
                              int min_periods):
    cdef int n = x.shape[0]
    if n < 2:
        return (np.nan, np.nan)
    cdef double best_error = INFINITY
    cdef double candidate_slope, candidate_intercept
    cdef double y_pred, res, median_resid, thresh, inlier_error
    cdef int i, j, idx1, idx2, best_inlier_count = 0
    cdef list best_inliers = []
    cdef list current_inliers
    cdef np.ndarray[DTYPE_t, ndim=1] residuals = np.empty(n, dtype=np.float64)
    cdef double best_slope = np.nan, best_intercept = np.nan

    for i in range(ransac_iterations):
        current_inliers = []
        # randomly select 2 distinct indices
        idx1 = random.randrange(n)
        idx2 = random.randrange(n)
        while idx2 == idx1:
            idx2 = random.randrange(n)
        if x[idx2] == x[idx1]:
            continue
        candidate_slope = (y[idx2] - y[idx1]) / (x[idx2] - x[idx1])
        candidate_intercept = y[idx1] - candidate_slope * x[idx1]
        # Compute residuals
        for j in range(n):
            y_pred = candidate_slope * x[j] + candidate_intercept
            res = fabs(y[j] - y_pred)
            residuals[j] = res
        median_resid = _median(residuals)
        thresh = ransac_threshold * (median_resid if median_resid > 0 else 1.0)
        for j in range(n):
            if residuals[j] < thresh:
                current_inliers.append(j)
        if len(current_inliers) < max(2, min_periods):
            continue
        inlier_error = 0.0
        for j in current_inliers:
            inlier_error += weights[j] * residuals[j]
        if inlier_error < best_error:
            best_error = inlier_error
            best_inlier_count = len(current_inliers)
            best_inliers = current_inliers[:]
            best_slope = candidate_slope
            best_intercept = candidate_intercept

    if best_inlier_count < max(2, min_periods):
        return (np.nan, np.nan)

    # Refit using only best inliers.
    cdef int m = best_inlier_count
    cdef np.ndarray[DTYPE_t, ndim=1] x_inliers = np.empty(m, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] y_inliers = np.empty(m, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] weights_inliers = np.empty(m, dtype=np.float64)
    for i in range(m):
        x_inliers[i] = x[best_inliers[i]]
        y_inliers[i] = y[best_inliers[i]]
        weights_inliers[i] = weights[best_inliers[i]]
    return weighted_linear_regression(x_inliers, y_inliers, weights_inliers)

# cdef function for recursive mode regression.
cdef tuple _regress_recursive(np.ndarray[DTYPE_t, ndim=1] x,
                              np.ndarray[DTYPE_t, ndim=1] y,
                              double alpha,
                              int min_periods):
    cdef int n = x.shape[0], t, m, i
    cdef np.ndarray[DTYPE_t, ndim=1] slopes = np.full(n, np.nan, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] intercepts = np.full(n, np.nan, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] r2s = np.full(n, np.nan, dtype=np.float64)
    cdef double S0 = 1.0, S1 = x[0], S2 = x[0]*x[0]
    cdef double Sy = y[0], Sxy = x[0]*y[0], Syy = y[0]*y[0]
    cdef double mean_x, mean_y, var_x, cov, slope, intercept, TSS, RSS

    slopes[0] = np.nan
    intercepts[0] = np.nan
    r2s[0] = np.nan

    for t in range(1, n):
        S0 = (1 - alpha) * S0 + 1.0
        S1 = (1 - alpha) * S1 + x[t]
        S2 = (1 - alpha) * S2 + x[t]*x[t]
        Sy = (1 - alpha) * Sy + y[t]
        Sxy = (1 - alpha) * Sxy + x[t]*y[t]
        Syy = (1 - alpha) * Syy + y[t]*y[t]
        if t + 1 < min_periods:
            slopes[t] = np.nan
            intercepts[t] = np.nan
            r2s[t] = np.nan
            continue
        mean_x = S1 / S0
        mean_y = Sy / S0
        var_x = S2 / S0 - mean_x*mean_x
        cov = Sxy / S0 - mean_x*mean_y
        if var_x > 0:
            slope = cov / var_x
            intercept = mean_y - slope*mean_x
        else:
            slope = np.nan
            intercept = np.nan
        slopes[t] = slope
        intercepts[t] = intercept
        TSS = Syy/S0 - mean_y*mean_y
        RSS = (Syy/S0) - 2*(slope*(Sxy/S0) + intercept*(Sy/S0)) + (slope*slope*(S2/S0) + 2*slope*intercept*(S1/S0) + intercept*intercept)
        if TSS > 0:
            r2s[t] = 1 - (RSS/TSS)
        else:
            r2s[t] = np.nan
    return slopes, intercepts, r2s

# The main class.
cdef class LinearRegression:
    cdef double alpha
    cdef bint adjust
    cdef int min_periods
    cdef object outlier_method  # string or None
    cdef int ransac_iterations
    cdef double ransac_threshold

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

    cpdef tuple regress(self, np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] y):
        cdef int n = x.shape[0], t, m, i
        cdef np.ndarray[DTYPE_t, ndim=1] slopes = np.full(n, np.nan, dtype=np.float64)
        cdef np.ndarray[DTYPE_t, ndim=1] intercepts = np.full(n, np.nan, dtype=np.float64)
        cdef np.ndarray[DTYPE_t, ndim=1] r2s = np.full(n, np.nan, dtype=np.float64)
        cdef np.ndarray[DTYPE_t, ndim=1] weights, x_sub, y_sub
        cdef double slope, intercept, r2_val

        if self.adjust:
            # Explicit mode: for each t, use data[0:t+1].
            for t in range(n):
                if t + 1 < self.min_periods:
                    continue
                m = t + 1
                weights = np.empty(m, dtype=np.float64)
                for i in range(m):
                    weights[i] = (1 - self.alpha) ** (m - 1 - i)
                x_sub = x[:m].copy()
                y_sub = y[:m].copy()
                if self.outlier_method == "ransac":
                    slope, intercept = ransac_regression(x_sub, y_sub, weights,
                                                         self.ransac_iterations,
                                                         self.ransac_threshold,
                                                         self.min_periods)
                else:
                    slope, intercept = weighted_linear_regression(x_sub, y_sub, weights)
                slopes[t] = slope
                intercepts[t] = intercept
                if not isnan(slope) and not isnan(intercept):
                    r2_val = weighted_r2(x_sub, y_sub, slope, intercept, weights)
                    r2s[t] = r2_val
        else:
            slopes, intercepts, r2s = _regress_recursive(x, y, self.alpha, self.min_periods)
        return slopes, intercepts, r2s

