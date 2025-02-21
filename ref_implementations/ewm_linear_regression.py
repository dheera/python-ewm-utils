import numpy as np

class EWMLinearRegression:
    def __init__(self, alpha, adjust=True, min_periods=1,
                 outlier_method=None, ransac_iterations=100, ransac_threshold=2.0):
        """
        Parameters:
          alpha : float
            Smoothing factor (0 < alpha <= 1).
          adjust : bool, default True
            When True, explicit weights are computed for all observations (like pandas).
            When False, a recursive update is used (robust outlier rejection is ignored in this mode).
          min_periods : int, default 1
            Minimum number of observations required for a valid result.
          outlier_method : str or None, default None
            Specify 'ransac' for robust outlier rejection using a simple RANSAC procedure.
            If None, no outlier rejection is performed.
          ransac_iterations : int, default 100
            Number of iterations to run the RANSAC procedure.
          ransac_threshold : float, default 2.0
            Scaling factor applied to the median absolute residual error to determine the inlier threshold.
        """
        if not (0 < alpha <= 1):
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.adjust = adjust
        self.min_periods = min_periods
        self.outlier_method = outlier_method
        self.ransac_iterations = ransac_iterations
        self.ransac_threshold = ransac_threshold

    def _weighted_linear_regression(self, x, y, weights):
        """Compute weighted least squares estimates for slope and intercept."""
        sum_w = weights.sum()
        mean_x = np.sum(x * weights) / sum_w
        mean_y = np.sum(y * weights) / sum_w
        cov = np.sum(weights * (x - mean_x) * (y - mean_y))
        var_x = np.sum(weights * (x - mean_x)**2)
        if var_x == 0:
            return np.nan, np.nan
        slope = cov / var_x
        intercept = mean_y - slope * mean_x
        return slope, intercept

    def _weighted_r2(self, x, y, slope, intercept, weights):
        """
        Compute the weighted R^2 given data, a model (slope & intercept),
        and weights. Here the weighted total sum-of-squares is computed as
          TSS = sum(w*(y - y_mean)^2)
        and the residual sum-of-squares is
          RSS = sum(w*(y - (slope*x+intercept))^2).
        """
        sum_w = weights.sum()
        if sum_w == 0:
            return np.nan
        y_mean = np.sum(y*weights)/sum_w
        TSS = np.sum(weights*(y - y_mean)**2)
        RSS = np.sum(weights*(y - (slope*x + intercept))**2)
        return 1 - (RSS/TSS) if TSS > 0 else np.nan

    def _ransac_regression(self, x, y, weights):
        """
        A simple RANSAC procedure that repeatedly selects 2 random points,
        computes the line, and then finds inliers based on the residuals.
        """
        n = len(x)
        if n < 2:
            return np.nan, np.nan
        best_error = np.inf
        best_inlier_mask = None
        for _ in range(self.ransac_iterations):
            idx = np.random.choice(n, 2, replace=False)
            x_sample = x[idx]
            y_sample = y[idx]
            if x_sample[1] == x_sample[0]:
                continue
            candidate_slope = (y_sample[1] - y_sample[0]) / (x_sample[1] - x_sample[0])
            candidate_intercept = y_sample[0] - candidate_slope * x_sample[0]
            y_pred = candidate_slope * x + candidate_intercept
            residuals = np.abs(y - y_pred)
            median_resid = np.median(residuals)
            thresh = self.ransac_threshold * (median_resid if median_resid > 0 else 1.0)
            inlier_mask = residuals < thresh
            if inlier_mask.sum() < max(2, self.min_periods):
                continue
            inlier_error = np.sum(weights[inlier_mask] * residuals[inlier_mask])
            if inlier_error < best_error:
                best_error = inlier_error
                best_inlier_mask = inlier_mask

        if best_inlier_mask is not None and best_inlier_mask.sum() >= max(2, self.min_periods):
            slope, intercept = self._weighted_linear_regression(x[best_inlier_mask],
                                                                y[best_inlier_mask],
                                                                weights[best_inlier_mask])
            return slope, intercept
        else:
            return np.nan, np.nan

    def regress(self, x, y):
        """
        Compute the EWM linear regression estimates (slope, intercept, and R^2) for each time step.

        Parameters:
          x, y : array_like (1D)
            The independent and dependent variable series (must have the same length).

        Returns:
          slopes, intercepts, r2s : tuple of np.ndarray
            Arrays containing the slope, intercept, and R^2 estimates for each time point.
            For indices with fewer than min_periods observations, np.nan is returned.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("Only 1D arrays are supported.")
        if len(x) != len(y):
            raise ValueError("x and y must be the same length.")

        n = len(x)
        slopes = np.full(n, np.nan)
        intercepts = np.full(n, np.nan)
        r2s = np.full(n, np.nan)

        if self.adjust:
            # Explicit mode: compute regression on data [0...t] for each t.
            for t in range(n):
                if t + 1 < self.min_periods:
                    continue
                weights = (1 - self.alpha) ** np.arange(t, -1, -1)
                x_sub = x[:t+1]
                y_sub = y[:t+1]
                if self.outlier_method == 'ransac':
                    slope, intercept = self._ransac_regression(x_sub, y_sub, weights)
                else:
                    slope, intercept = self._weighted_linear_regression(x_sub, y_sub, weights)
                slopes[t] = slope
                intercepts[t] = intercept
                if not np.isnan(slope) and not np.isnan(intercept):
                    r2s[t] = self._weighted_r2(x_sub, y_sub, slope, intercept, weights)
        else:
            # Recursive mode: update weighted sums.
            S0 = 1.0       # sum of weights
            S1 = x[0]      # weighted sum of x
            S2 = x[0]**2   # weighted sum of x^2
            Sy = y[0]      # weighted sum of y
            Sxy = x[0]*y[0]  # weighted sum of x*y
            Syy = y[0]**2     # weighted sum of y^2
            slopes[0] = np.nan
            intercepts[0] = np.nan
            r2s[0] = np.nan

            for t in range(1, n):
                S0 = (1 - self.alpha) * S0 + 1.0
                S1 = (1 - self.alpha) * S1 + x[t]
                S2 = (1 - self.alpha) * S2 + x[t]**2
                Sy = (1 - self.alpha) * Sy + y[t]
                Sxy = (1 - self.alpha) * Sxy + x[t]*y[t]
                Syy = (1 - self.alpha) * Syy + y[t]**2

                if t + 1 < self.min_periods:
                    slopes[t] = np.nan
                    intercepts[t] = np.nan
                    r2s[t] = np.nan
                    continue

                mean_x = S1 / S0
                mean_y = Sy / S0
                var_x = S2 / S0 - mean_x**2
                cov = Sxy / S0 - mean_x * mean_y
                if var_x > 0:
                    slope = cov / var_x
                    intercept = mean_y - slope * mean_x
                else:
                    slope, intercept = np.nan, np.nan

                slopes[t] = slope
                intercepts[t] = intercept

                # Compute weighted TSS and RSS using the aggregated sums:
                # TSS = (Syy/S0) - (Sy/S0)^2
                TSS = Syy/S0 - mean_y**2
                # RSS = (Syy/S0) - 2*(slope*(Sxy/S0)+intercept*(Sy/S0)) + (slope**2*(S2/S0) + 2*slope*intercept*(S1/S0) + intercept**2)
                RSS = (Syy/S0) - 2*(slope*(Sxy/S0) + intercept*(Sy/S0)) + (slope**2*(S2/S0) + 2*slope*intercept*(S1/S0) + intercept**2)
                r2s[t] = 1 - (RSS/TSS) if TSS > 0 else np.nan

        return slopes, intercepts, r2s

# Example usage:
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Generate synthetic data with a linear trend plus noise and outliers.
    np.random.seed(0)
    n_points = 200
    x = np.linspace(0, 10, n_points)
    true_slope = 2.0
    true_intercept = 1.0
    noise = np.random.randn(n_points) * 1.0
    y = true_slope * x + true_intercept + noise

    # Introduce some outliers.
    outlier_indices = np.random.choice(n_points, size=10, replace=False)
    y[outlier_indices] += np.random.randn(10) * 20

    # Create EWM regression objects.
    # 1. Standard weighted regression (explicit mode, no outlier rejection)
    ewm_lr_standard = EWMLinearRegression(alpha=0.05, adjust=True, min_periods=20)
    slopes_std, intercepts_std, r2_std = ewm_lr_standard.regress(x, y)

    # 2. Robust regression using RANSAC
    import time
    t = time.time()
    ewm_lr_ransac = EWMLinearRegression(alpha=0.05, adjust=True, min_periods=20,
                                          outlier_method='ransac', ransac_iterations=200, ransac_threshold=2.5)
    slopes_ransac, intercepts_ransac, r2_ransac = ewm_lr_ransac.regress(x, y)
    print(time.time() - t)

    # Plot the evolution of slope estimates.
    plt.figure(figsize=(12, 5))
    plt.plot(x, slopes_std, label='Standard EWM Slope', color='blue')
    plt.plot(x, slopes_ransac, label='Robust EWM Slope (RANSAC)', color='red')
    plt.axhline(true_slope, color='black', linestyle='--', label='True Slope')
    plt.xlabel('x')
    plt.ylabel('Slope')
    plt.legend()
    plt.title('EWM Linear Regression Slope Estimates')
    plt.show()

    # Plot the evolution of R².
    plt.figure(figsize=(12, 5))
    plt.plot(x, r2_std, label='Standard EWM R²', color='blue')
    plt.plot(x, r2_ransac, label='Robust EWM R² (RANSAC)', color='red')
    plt.xlabel('x')
    plt.ylabel('R²')
    plt.legend()
    plt.title('EWM Linear Regression R² Estimates')
    plt.show()

    # Plot the regression lines at the final time point.
    final_std = slopes_std[-1] * x + intercepts_std[-1]
    final_ransac = slopes_ransac[-1] * x + intercepts_ransac[-1]
    plt.figure(figsize=(12, 5))
    plt.scatter(x, y, label='Data', s=10, alpha=0.6)
    plt.plot(x, final_std, label='Standard EWM Regression', color='blue', linewidth=2)
    plt.plot(x, final_ransac, label='Robust EWM Regression (RANSAC)', color='red', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Final Regression Lines from EWM Linear Regression')
    plt.show()
