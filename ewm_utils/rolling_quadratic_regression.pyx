# ewm_rolling_quadratic_regression.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
cimport numpy as np
from libc.math cimport fabs, isnan, INFINITY
import random

ctypedef np.double_t DTYPE_t

###############################################################################
#             Helper functions for quadratic regression                     #
###############################################################################

cdef tuple _weighted_quadratic_regression(np.ndarray[DTYPE_t, ndim=1] x,
                                            np.ndarray[DTYPE_t, ndim=1] y,
                                            np.ndarray[DTYPE_t, ndim=1] weights):
    """
    Compute weighted least squares estimates for the quadratic model:
         y = a*x^2 + b*x + c.
    Returns (a, b, c).
    """
    cdef np.ndarray X, W, Xw, yw, beta
    X = np.vstack((x*x, x, np.ones_like(x))).T
    W = np.sqrt(weights)
    Xw = X * W[:, None]
    yw = y * W
    try:
        beta, residuals, rank, s = np.linalg.lstsq(Xw, yw, rcond=None)
    except np.linalg.LinAlgError:
        return (np.nan, np.nan, np.nan)
    return (beta[0], beta[1], beta[2])

cdef double _weighted_r2_quadratic(np.ndarray[DTYPE_t, ndim=1] x,
                                   np.ndarray[DTYPE_t, ndim=1] y,
                                   double a, double b, double c,
                                   np.ndarray[DTYPE_t, ndim=1] weights):
    """
    Compute the weighted R² for the quadratic model.
    """
    cdef int n = x.shape[0]
    cdef double sum_w = 0.0, mean_y = 0.0, TSS = 0.0, RSS = 0.0, r2, y_pred
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
        y_pred = a*x[i]*x[i] + b*x[i] + c
        RSS += weights[i] * (y[i] - y_pred) * (y[i] - y_pred)
    if TSS > 0:
        r2 = 1 - RSS/TSS
    else:
        r2 = np.nan
    return r2

cdef tuple _ransac_regression_quadratic(np.ndarray[DTYPE_t, ndim=1] x,
                                          np.ndarray[DTYPE_t, ndim=1] y,
                                          np.ndarray[DTYPE_t, ndim=1] weights,
                                          int ransac_iterations,
                                          double ransac_threshold,
                                          int min_periods):
    """
    Perform a simple RANSAC procedure for the quadratic model.
    Randomly selects 3 distinct points to form a candidate model.
    Returns (a, b, c) of the best candidate or (np.nan, np.nan, np.nan).
    """
    cdef int n = x.shape[0]
    if n < 3:
        return (np.nan, np.nan, np.nan)
    cdef double best_error = INFINITY
    cdef double candidate_a, candidate_b, candidate_c
    cdef double y_pred, res, median_resid, thresh, inlier_error
    cdef int i, j, inlier_count
    cdef np.ndarray residuals = np.empty(n, dtype=np.float64)
    cdef np.ndarray best_mask = None
    cdef np.ndarray current_mask
    cdef int idx0, idx1, idx2
    cdef np.ndarray[DTYPE_t, ndim=1] x_sample, y_sample
    cdef np.ndarray[DTYPE_t, ndim=2] X_sample
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
        # Manually build small arrays for the sample.
        x_sample = np.empty(3, dtype=np.float64)
        y_sample = np.empty(3, dtype=np.float64)
        x_sample[0] = x[idx0]
        x_sample[1] = x[idx1]
        x_sample[2] = x[idx2]
        y_sample[0] = y[idx0]
        y_sample[1] = y[idx1]
        y_sample[2] = y[idx2]
        X_sample = np.empty((3, 3), dtype=np.float64)
        for j in range(3):
            X_sample[j, 0] = x_sample[j]*x_sample[j]
            X_sample[j, 1] = x_sample[j]
            X_sample[j, 2] = 1.0
        try:
            beta = np.linalg.solve(X_sample, y_sample)
        except np.linalg.LinAlgError:
            continue
        candidate_a = beta[0]
        candidate_b = beta[1]
        candidate_c = beta[2]
        for j in range(n):
            y_pred = candidate_a * x[j]*x[j] + candidate_b * x[j] + candidate_c
            res = fabs(y[j] - y_pred)
            residuals[j] = res
        median_resid = np.median(residuals)
        thresh = ransac_threshold * (median_resid if median_resid > 0 else 1.0)
        current_mask = (residuals < thresh)
        inlier_count = np.sum(current_mask)
        if inlier_count < max(3, min_periods):
            continue
        inlier_error = np.sum(weights[current_mask] * residuals[current_mask])
        if inlier_error < best_error:
            best_error = inlier_error
            best_mask = current_mask.copy()
    if best_mask is not None and np.sum(best_mask) >= max(3, min_periods):
        return _weighted_quadratic_regression(x[best_mask], y[best_mask], weights[best_mask])
    else:
        return (np.nan, np.nan, np.nan)

###############################################################################
#             Rolling  Quadratic Regression Class                        #
###############################################################################

cdef class RollingQuadraticRegression:
    cdef public double alpha
    cdef public bint adjust
    cdef public int min_periods
    cdef public object outlier_method   # str or None
    cdef public int ransac_iterations
    cdef public double ransac_threshold

    # Explicit mode: store all data.
    cdef list xs
    cdef list ys

    # Recursive mode: weighted sums for quadratic regression.
    cdef public int count
    cdef double S0, S1, S2, S3, S4, Sy, Sxy, Sx2y, Syy

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
            self.S3 = 0.0
            self.S4 = 0.0
            self.Sy = 0.0
            self.Sxy = 0.0
            self.Sx2y = 0.0
            self.Syy = 0.0

    cpdef tuple update(self, double x, double y):
        """
        Update the regression with a new (x, y) data point.
        Returns (a, b, c, r2) where the model is y = a*x^2 + b*x + c.
        If there are fewer than min_periods points, returns NaNs.
        """
        cdef np.ndarray weights, x_arr, y_arr
        cdef int n, i
        cdef double a, b, c, r2, mean_y, TSS, RSS
        if self.adjust:
            self.xs.append(x)
            self.ys.append(y)
            n = len(self.xs)
            if n < self.min_periods:
                return (np.nan, np.nan, np.nan, np.nan)
            weights = np.empty(n, dtype=np.float64)
            for i in range(n):
                weights[i] = (1 - self.alpha) ** (n - 1 - i)
            x_arr = np.array(self.xs, dtype=np.float64)
            y_arr = np.array(self.ys, dtype=np.float64)
            if self.outlier_method == 'ransac':
                a, b, c = _ransac_regression_quadratic(x_arr, y_arr, weights,
                                                       self.ransac_iterations,
                                                       self.ransac_threshold,
                                                       self.min_periods)
            else:
                a, b, c = _weighted_quadratic_regression(x_arr, y_arr, weights)
            r2 = _weighted_r2_quadratic(x_arr, y_arr, a, b, c, weights)
            return (a, b, c, r2)
        else:
            # Recursive mode.
            self.count += 1
            if self.count == 1:
                self.S0 = 1.0
                self.S1 = x
                self.S2 = x**2
                self.S3 = x**3
                self.S4 = x**4
                self.Sy = y
                self.Sxy = x*y
                self.Sx2y = (x**2)*y
                self.Syy = y**2
                return (np.nan, np.nan, np.nan, np.nan)
            else:
                self.S0 = (1 - self.alpha) * self.S0 + 1.0
                self.S1 = (1 - self.alpha) * self.S1 + x
                self.S2 = (1 - self.alpha) * self.S2 + x**2
                self.S3 = (1 - self.alpha) * self.S3 + x**3
                self.S4 = (1 - self.alpha) * self.S4 + x**4
                self.Sy = (1 - self.alpha) * self.Sy + y
                self.Sxy = (1 - self.alpha) * self.Sxy + x*y
                self.Sx2y = (1 - self.alpha) * self.Sx2y + (x**2)*y
                self.Syy = (1 - self.alpha) * self.Syy + y**2
                if self.count < self.min_periods:
                    return (np.nan, np.nan, np.nan, np.nan)
                # Solve normal equations: M * [a,b,c] = B.
                M = np.array([[self.S4, self.S3, self.S2],
                              [self.S3, self.S2, self.S1],
                              [self.S2, self.S1, self.S0]], dtype=np.float64)
                B = np.array([self.Sx2y, self.Sxy, self.Sy], dtype=np.float64)
                try:
                    beta = np.linalg.solve(M, B)
                    a = beta[0]
                    b = beta[1]
                    c = beta[2]
                except np.linalg.LinAlgError:
                    a = b = c = np.nan
                mean_y = self.Sy / self.S0
                TSS = self.Syy / self.S0 - mean_y**2
                RSS = (self.Syy / self.S0) - 2*((a*(self.Sx2y/self.S0)) + (b*(self.Sxy/self.S0)) + (c*(self.Sy/self.S0))) \
                      + (a*a*(self.S4/self.S0) + 2*a*b*(self.S3/self.S0) + 2*a*c*(self.S2/self.S0) + b*b*(self.S2/self.S0) + 2*b*c*(self.S1/self.S0) + c*c)
                if TSS > 0:
                    r2 = 1 - RSS / TSS
                else:
                    r2 = np.nan
                return (a, b, c, r2)

