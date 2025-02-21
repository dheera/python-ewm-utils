import numpy as np
from ewm_utils import RollingLinearRegression
import time

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Generate synthetic data with a linear trend plus noise and outliers.
    np.random.seed(0)
    n_points = 200
    x = np.linspace(0, 10, n_points)
    true_slope = 2.0
    true_intercept = 1.0
    noise = np.random.randn(n_points)
    y = true_slope * x + true_intercept + noise

    # Introduce some outliers.
    outlier_indices = np.random.choice(n_points, size=10, replace=False)
    y[outlier_indices] += np.random.randn(10) * 20

    slopes = []
    intercepts = []
    r2_values = []

    ewmlr = RollingLinearRegression(alpha=0.05, adjust=True, min_periods=20,
                                       outlier_method='ransac', ransac_iterations=200, ransac_threshold=2.5)
    for xi, yi in zip(x, y):
        t = time.time()
        slope, intercept, r2 = ewmlr.update(xi, yi)
        print(f"RANSAC quadratic rolling update took {time.time() - t} sec")
        slopes.append(slope)
        intercepts.append(intercept)
        r2_values.append(r2)

    # Plot evolving slope estimates.
    plt.figure(figsize=(12, 5))
    plt.plot(x, slopes, label='Rolling  Slope (RANSAC)', color='red')
    plt.axhline(true_slope, color='black', linestyle='--', label='True Slope')
    plt.xlabel('x')
    plt.ylabel('Slope')
    plt.legend()
    plt.title('Rolling  Linear Regression: Slope Estimates')
    plt.show()

    # Plot evolving R² values.
    plt.figure(figsize=(12, 5))
    plt.plot(x, r2_values, label='Rolling  R²', color='green')
    plt.xlabel('x')
    plt.ylabel('R²')
    plt.legend()
    plt.title('Rolling  Linear Regression: R² Estimates')
    plt.show()

    # Plot final regression line.
    final_slope = slopes[-1]
    final_intercept = intercepts[-1]
    plt.figure(figsize=(12, 5))
    plt.scatter(x, y, label='Data', s=10, alpha=0.6)
    plt.plot(x, final_slope * x + final_intercept, label='Final Regression', color='red', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Final Regression Line from Rolling  Linear Regression')
    plt.show()
