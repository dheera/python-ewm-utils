import time
import numpy as np
from ewm_utils import QuadraticRegression
import matplotlib.pyplot as plt

# Generate synthetic data with a quadratic trend plus noise.
np.random.seed(0)
n_points = 200
xs = np.linspace(-10, 10, n_points)
true_a = 0.5
true_b = -2.0
true_c = 3.0
noise = np.random.randn(n_points) * 5.0
ys = true_a * xs**2 + true_b * xs + true_c + noise

# Introduce some outliers.
outlier_indices = np.random.choice(n_points, size=10, replace=False)
ys[outlier_indices] += np.random.randn(10) * 30

# Create quadratic regression objects.
# 1. Standard weighted regression (explicit mode, no outlier rejection)
t = time.time()
ewm_quad_standard = QuadraticRegression(alpha=0.05, adjust=True, min_periods=20)
a_std, b_std, c_std, r2 = ewm_quad_standard.regress(xs, ys)
print("Standard execution time:", time.time() - t)

# 2. Robust regression using RANSAC (explicit mode)
t = time.time()
ewm_quad_ransac = QuadraticRegression(alpha=0.05, adjust=True, min_periods=20,
                                          outlier_method='ransac', ransac_iterations=200, ransac_threshold=2.5)
a_ransac, b_ransac, c_ransac, r2 = ewm_quad_ransac.regress(xs, ys)
print("RANSAC execution time:", time.time() - t)

# Plot evolving coefficient estimates.
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(xs, a_std, label='Standard a', color='blue')
plt.plot(xs, a_ransac, label='Robust a (RANSAC)', color='red')
plt.axhline(true_a, color='black', linestyle='--', label='True a')
plt.ylabel('Coefficient a')
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(xs, b_std, label='Standard b', color='blue')
plt.plot(xs, b_ransac, label='Robust b (RANSAC)', color='red')
plt.axhline(true_b, color='black', linestyle='--', label='True b')
plt.ylabel('Coefficient b')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(xs, c_std, label='Standard c', color='blue')
plt.plot(xs, c_ransac, label='Robust c (RANSAC)', color='red')
plt.axhline(true_c, color='black', linestyle='--', label='True c')
plt.xlabel('x')
plt.ylabel('Coefficient c')
plt.legend()
plt.suptitle(' Quadratic Regression Coefficient Estimates')
plt.show()

# Plot the final regression curves.
final_a_std = a_std[-1]
final_b_std = b_std[-1]
final_c_std = c_std[-1]
final_a_ransac = a_ransac[-1]
final_b_ransac = b_ransac[-1]
final_c_ransac = c_ransac[-1]

xs_fit = np.linspace(min(xs), max(xs), 300)
ys_fit_std = final_a_std * xs_fit**2 + final_b_std * xs_fit + final_c_std
ys_fit_ransac = final_a_ransac * xs_fit**2 + final_b_ransac * xs_fit + final_c_ransac

plt.figure(figsize=(12, 5))
plt.scatter(xs, ys, label='Data', s=10, alpha=0.6)
plt.plot(xs_fit, ys_fit_std, label='Standard Regression', color='blue', linewidth=2)
plt.plot(xs_fit, ys_fit_ransac, label='Robust Regression (RANSAC)', color='red', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Final Quadratic Regression Curves')
plt.show()


