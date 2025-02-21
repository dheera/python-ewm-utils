# ewm_quadratic_regression.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
cimport numpy as np
from libc.math cimport fabs, isnan, INFINITY
import random

ctypedef np.double_t DTYPE_t

###############################################################################
#                    QUADRATIC REGRESSION HELPER FUNCTIONS                    #
###############################################################################

# Standard weighted quadratic regression.
# Solves for beta in y = a*x^2 + b*x + c via weighted least squares.
cpdef tuple weighted_quadratic_regression(np.ndarray[DTYPE_t, ndim=1] x,
                                            np.ndarray[DTYPE_t, ndim=1] y,
                                            np.ndarray[DTYPE_t, ndim=1] weights):
    """
    Returns (a, b, c) solving the weighted LS problem.
    """
    cdef np.ndarray X, W, Xw, yw, beta
    # Construct design matrix with columns: x^2, x, 1.
    X = np.vstack((x*x, x, np.ones_like(x))).T
    # Apply weights (as sqrt of weights)
    W = np.sqrt(weights)
    Xw = X * W[:, None]
    yw = y * W
    try:
        beta, residuals, rank, s = np.linalg.lstsq(Xw, yw, rcond=None)
    except np.linalg.LinAlgError:
        return (np.nan, np.nan, np.nan)
    return (beta[0], beta[1], beta[2])

# Robust quadratic regression using RANSAC.
cpdef tuple ransac_quadratic_regression(np.ndarray[DTYPE_t, ndim=1] x,
                                          np.ndarray[DTYPE_t, ndim=1] y,
                                          np.ndarray[DTYPE_t, ndim=1] weights,
                                          int ransac_iterations,
                                          double ransac_threshold,
                                          int min_periods):
    """
    Performs a RANSAC procedure for the quadratic model.
    Randomly selects 3 distinct points, solves the 3x3 system,
    then evaluates all points for inliers.
    Returns (a, b, c) if a candidate with enough inliers is found; otherwise, NaNs.
    """
    cdef int n = x.shape[0]
    if n < 3:
        return (np.nan, np.nan, np.nan)
    cdef double best_error = INFINITY
    cdef double candidate_a, candidate_b, candidate_c
    cdef double y_pred, res, median_resid, thresh, inlier_error
    cdef int i, j, idx0, idx1, idx2, best_inlier_count = 0
    cdef list best_inliers = []
    cdef list current_inliers
    cdef np.ndarray residuals = np.empty(n, dtype=np.float64)
    cdef double best_a = np.nan, best_b = np.nan, best_c = np.nan
    # Declare variables to be used inside the loop.
    cdef np.ndarray[DTYPE_t, ndim=1] x_sample, y_sample
    cdef np.ndarray[DTYPE_t, ndim=2] X_sample

    for i in range(ransac_iterations):
        current_inliers = []
        # Randomly sample 3 distinct indices.
        idx0 = random.randrange(n)
        idx1 = random.randrange(n)
        while idx1 == idx0:
            idx1 = random.randrange(n)
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
        # Build the design matrix X_sample with columns: x^2, x, 1.
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
        for j in range(n):
            if residuals[j] < thresh:
                current_inliers.append(j)
        if len(current_inliers) < max(3, min_periods):
            continue
        inlier_error = 0.0
        for j in current_inliers:
            inlier_error += weights[j] * residuals[j]
        if inlier_error < best_error:
            best_error = inlier_error
            best_inlier_count = len(current_inliers)
            best_inliers = current_inliers[:]
            best_a = candidate_a
            best_b = candidate_b
            best_c = candidate_c

    if best_inlier_count < max(3, min_periods):
        return (np.nan, np.nan, np.nan)
    cdef int m = best_inlier_count
    cdef np.ndarray[DTYPE_t, ndim=1] x_inliers = np.empty(m, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] y_inliers = np.empty(m, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] weights_inliers = np.empty(m, dtype=np.float64)
    for i in range(m):
        x_inliers[i] = x[best_inliers[i]]
        y_inliers[i] = y[best_inliers[i]]
        weights_inliers[i] = weights[best_inliers[i]]
    return weighted_quadratic_regression(x_inliers, y_inliers, weights_inliers)

# Compute weighted R² for the quadratic model.
cpdef double weighted_r2_quadratic(np.ndarray[DTYPE_t, ndim=1] x,
                                   np.ndarray[DTYPE_t, ndim=1] y,
                                   double a, double b, double c,
                                   np.ndarray[DTYPE_t, ndim=1] weights):
    cdef int n = x.shape[0]
    cdef double sum_w = 0.0, y_mean = 0.0, TSS = 0.0, RSS = 0.0, y_pred
    cdef int i
    for i in range(n):
        sum_w += weights[i]
    if sum_w == 0:
        return np.nan
    for i in range(n):
        y_mean += y[i] * weights[i]
    y_mean /= sum_w
    for i in range(n):
        TSS += weights[i] * (y[i] - y_mean)*(y[i] - y_mean)
        y_pred = a*x[i]*x[i] + b*x[i] + c
        RSS += weights[i] * (y[i] - y_pred)*(y[i] - y_pred)
    if TSS > 0:
        return 1 - RSS/TSS
    else:
        return np.nan

# Recursive mode helper for quadratic regression.
cdef tuple _regress_quadratic_recursive(np.ndarray[DTYPE_t, ndim=1] x,
                                          np.ndarray[DTYPE_t, ndim=1] y,
                                          double alpha,
                                          int min_periods):
    cdef int n = x.shape[0], t, i
    cdef np.ndarray[DTYPE_t, ndim=1] a_coeffs = np.full(n, np.nan, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] b_coeffs = np.full(n, np.nan, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] c_coeffs = np.full(n, np.nan, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] r2s = np.full(n, np.nan, dtype=np.float64)
    cdef double S0, S1, S2, S3, S4, Sy, Sxy, Sx2y, Syy
    cdef double a, b, c, y_mean, TSS, RSS
    # Initialize with first observation.
    S0 = 1.0
    S1 = x[0]
    S2 = x[0]*x[0]
    S3 = x[0]*x[0]*x[0]
    S4 = x[0]*x[0]*x[0]*x[0]
    Sy = y[0]
    Sxy = x[0]*y[0]
    Sx2y = x[0]*x[0]*y[0]
    Syy = y[0]*y[0]
    a_coeffs[0] = np.nan
    b_coeffs[0] = np.nan
    c_coeffs[0] = np.nan
    r2s[0] = np.nan

    for t in range(1, n):
        S0 = (1 - alpha) * S0 + 1.0
        S1 = (1 - alpha) * S1 + x[t]
        S2 = (1 - alpha) * S2 + x[t]*x[t]
        S3 = (1 - alpha) * S3 + x[t]*x[t]*x[t]
        S4 = (1 - alpha) * S4 + x[t]*x[t]*x[t]*x[t]
        Sy = (1 - alpha) * Sy + y[t]
        Sxy = (1 - alpha) * Sxy + x[t]*y[t]
        Sx2y = (1 - alpha) * Sx2y + x[t]*x[t]*y[t]
        Syy = (1 - alpha) * Syy + y[t]*y[t]
        if t + 1 < min_periods:
            continue
        # Build normal equations.
        # [ S4  S3  S2 ] [a] = [ Sx2y ]
        # [ S3  S2  S1 ] [b] = [ Sxy  ]
        # [ S2  S1  S0 ] [c] = [ Sy   ]
        M = np.array([[S4, S3, S2],
                      [S3, S2, S1],
                      [S2, S1, S0]], dtype=np.float64)
        B = np.array([Sx2y, Sxy, Sy], dtype=np.float64)
        try:
            beta = np.linalg.solve(M, B)
            a = beta[0]
            b = beta[1]
            c = beta[2]
        except np.linalg.LinAlgError:
            a = b = c = np.nan
        a_coeffs[t] = a
        b_coeffs[t] = b
        c_coeffs[t] = c
        y_mean = Sy / S0
        TSS = Syy/S0 - y_mean*y_mean
        RSS = (Syy/S0) - 2*((a*(Sx2y/S0)) + (b*(Sxy/S0)) + (c*(Sy/S0))) \
              + (a*a*(S4/S0) + 2*a*b*(S3/S0) + 2*a*c*(S2/S0) + b*b*(S2/S0) + 2*b*c*(S1/S0) + c*c)
        if TSS > 0:
            r2s[t] = 1 - (RSS/TSS)
        else:
            r2s[t] = np.nan

    return a_coeffs, b_coeffs, c_coeffs, r2s

###############################################################################
#                      QuadraticRegression CLASS                           #
###############################################################################

cdef class QuadraticRegression:
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
        """
        Compute the  quadratic regression estimates for each time step.
        Returns (a_coeffs, b_coeffs, c_coeffs, r2s). For time steps with fewer than
        min_periods observations, NaN is returned.
        """
        cdef int n = x.shape[0], t, m, i
        cdef np.ndarray[DTYPE_t, ndim=1] a_coeffs = np.full(n, np.nan, dtype=np.float64)
        cdef np.ndarray[DTYPE_t, ndim=1] b_coeffs = np.full(n, np.nan, dtype=np.float64)
        cdef np.ndarray[DTYPE_t, ndim=1] c_coeffs = np.full(n, np.nan, dtype=np.float64)
        cdef np.ndarray[DTYPE_t, ndim=1] r2s = np.full(n, np.nan, dtype=np.float64)
        cdef np.ndarray[DTYPE_t, ndim=1] weights, x_sub, y_sub
        cdef double a, b, c, r2_val

        if self.adjust:
            # Explicit mode: for each t, use data[0:t+1]
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
                    a, b, c = ransac_quadratic_regression(x_sub, y_sub, weights,
                                                           self.ransac_iterations,
                                                           self.ransac_threshold,
                                                           self.min_periods)
                else:
                    a, b, c = weighted_quadratic_regression(x_sub, y_sub, weights)
                a_coeffs[t] = a
                b_coeffs[t] = b
                c_coeffs[t] = c
                if not isnan(a) and not isnan(b) and not isnan(c):
                    r2_val = weighted_r2_quadratic(x_sub, y_sub, a, b, c, weights)
                    r2s[t] = r2_val
        else:
            a_coeffs, b_coeffs, c_coeffs, r2s = _regress_quadratic_recursive(x, y, self.alpha, self.min_periods)
        return a_coeffs, b_coeffs, c_coeffs, r2s

