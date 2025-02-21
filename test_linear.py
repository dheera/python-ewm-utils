import time
import numpy as np
from ewm_utils import LinearRegression
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

# Create  regression objects.
# 1. Standard weighted regression (explicit mode, no outlier rejection)
t = time.time()
ewm_lr_standard = LinearRegression(alpha=0.05, adjust=True, min_periods=20)
slopes_std, intercepts_std, r2_std = ewm_lr_standard.regress(x, y)
print("Standard execution time:", time.time() - t)

# 2. Robust regression using RANSAC
t = time.time()
ewm_lr_ransac = LinearRegression(alpha=0.05, adjust=True, min_periods=20,
                                      outlier_method='ransac', ransac_iterations=200, ransac_threshold=2.5)
slopes_ransac, intercepts_ransac, r2_ransac = ewm_lr_ransac.regress(x, y)
print("RANSAC execution time:", time.time() - t)

# Plot the evolution of slope estimates.
plt.figure(figsize=(12, 5))
plt.plot(x, slopes_std, label='Standard  Slope', color='blue')
plt.plot(x, slopes_ransac, label='Robust  Slope (RANSAC)', color='red')
plt.axhline(true_slope, color='black', linestyle='--', label='True Slope')
plt.xlabel('x')
plt.ylabel('Slope')
plt.legend()
plt.title(' Linear Regression Slope Estimates')
plt.show()

# Plot the evolution of R².
plt.figure(figsize=(12, 5))
plt.plot(x, r2_std, label='Standard  R²', color='blue')
plt.plot(x, r2_ransac, label='Robust  R² (RANSAC)', color='red')
plt.xlabel('x')
plt.ylabel('R²')
plt.legend()
plt.title(' Linear Regression R² Estimates')
plt.show()

# Plot the regression lines at the final time point.
final_std = slopes_std[-1] * x + intercepts_std[-1]
final_ransac = slopes_ransac[-1] * x + intercepts_ransac[-1]
plt.figure(figsize=(12, 5))
plt.scatter(x, y, label='Data', s=10, alpha=0.6)
plt.plot(x, final_std, label='Standard  Regression', color='blue', linewidth=2)
plt.plot(x, final_ransac, label='Robust  Regression (RANSAC)', color='red', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Final Regression Lines from  Linear Regression')
plt.show()

