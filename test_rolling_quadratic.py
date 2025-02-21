import numpy as np
from ewm_utils import RollingQuadraticRegression

# Example usage:
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Create a rolling  quadratic regression object.
    # For explicit mode with RANSAC robust regression.
    ewmlr = RollingQuadraticRegression(alpha=0.05, adjust=True, min_periods=20,
                                            outlier_method='ransac', ransac_iterations=200, ransac_threshold=2.5)

    # Generate synthetic data with a quadratic trend plus noise.
    np.random.seed(0)
    n_points = 200
    xs = np.linspace(-10, 10, n_points)
    # True quadratic: y = 0.5*x^2 - 2*x + 3.
    true_a = 0.5
    true_b = -2.0
    true_c = 3.0
    noise = np.random.randn(n_points) * 5.0
    ys = true_a * xs**2 + true_b * xs + true_c + noise

    # Introduce some outliers.
    outlier_indices = np.random.choice(n_points, size=10, replace=False)
    ys[outlier_indices] += np.random.randn(10) * 30

    a_list = []
    b_list = []
    c_list = []

    for xi, yi in zip(xs, ys):
        a, b, c, r2 = ewmlr.update(xi, yi)
        a_list.append(a)
        b_list.append(b)
        c_list.append(c)

    # Plot the evolving quadratic coefficient estimates.
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(xs, a_list, label='Estimate a', color='red')
    plt.axhline(true_a, color='black', linestyle='--', label='True a')
    plt.ylabel('Coefficient a')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(xs, b_list, label='Estimate b', color='blue')
    plt.axhline(true_b, color='black', linestyle='--', label='True b')
    plt.ylabel('Coefficient b')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(xs, c_list, label='Estimate c', color='green')
    plt.axhline(true_c, color='black', linestyle='--', label='True c')
    plt.xlabel('x')
    plt.ylabel('Coefficient c')
    plt.legend()
    plt.suptitle('Rolling  Quadratic Regression Coefficient Estimates')
    plt.show()

    # Plot the final regression curve.
    final_a = a_list[-1]
    final_b = b_list[-1]
    final_c = c_list[-1]
    plt.figure(figsize=(12, 5))
    plt.scatter(xs, ys, label='Data', s=10, alpha=0.6)
    xs_fit = np.linspace(min(xs), max(xs), 300)
    ys_fit = final_a * xs_fit**2 + final_b * xs_fit + final_c
    plt.plot(xs_fit, ys_fit, label='Final Regression', color='red', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Final Quadratic Regression Curve')
    plt.show()


