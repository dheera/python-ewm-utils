import numpy as np
from ewm_utils import RollingCorrelation
import unittest
import pandas as pd

# Unit test for rolling correlation that verifies against Pandas implementation

class TestRollingCorrelation(unittest.TestCase):
    def setUp(self):
        self.alpha = 0.1
        self.min_periods = 5
        np.random.seed(42)  # for reproducibility

    def _run_trial(self, adjust_flag):
        # Create a random data series of length between 50 and 150.
        n = np.random.randint(50, 151)
        x_vals = np.random.randn(n)
        # Create a correlated y: y = 0.5*x + noise.
        y_vals = 0.5 * x_vals + np.random.randn(n) * 0.5

        # Initialize our rolling  correlation.
        rolling_corr = RollingCorrelation(alpha=self.alpha, adjust=adjust_flag, min_periods=self.min_periods)

        # Collect rolling correlation results.
        rolling_results = []
        for i, (x, y) in enumerate(zip(x_vals, y_vals)):
            corr = rolling_corr.update(x, y)
            rolling_results.append(corr)

        rolling_results = np.array(rolling_results)

        # Now compute the expected results using pandas on each growing window.
        expected_results = []
        for i in range(n):
            if i + 1 < self.min_periods:
                expected_results.append(np.nan)
            else:
                # For each window, generate the appropriate weights.
                if adjust_flag:
                    # Explicit weighting:
                    window = pd.DataFrame({'x': x_vals[:i+1], 'y': y_vals[:i+1]})
                    # pandas uses the same weighting as we do (newest weight=1, oldest=(1-alpha)^(n-1))
                    expected_corr = window['x'].ewm(alpha=self.alpha, adjust=True,
                                                      min_periods=self.min_periods).corr(window['y']).iloc[-1]
                    expected_results.append(expected_corr)
                else:
                    # For recursive mode, pandas supports adjust=False.
                    window = pd.DataFrame({'x': x_vals[:i+1], 'y': y_vals[:i+1]})
                    expected_corr = window['x'].ewm(alpha=self.alpha, adjust=False,
                                                      min_periods=self.min_periods).corr(window['y']).iloc[-1]
                    expected_results.append(expected_corr)

        expected_results = np.array(expected_results)

        # Compare the rolling results with the expected results from pandas.
        np.testing.assert_allclose(rolling_results, expected_results, rtol=1e-6, equal_nan=True)

    def test_adjust_true_random_trials(self):
        # Run 5 random trials with adjust=True.
        for _ in range(5):
            self._run_trial(adjust_flag=True)

    def test_adjust_false_random_trials(self):
        # Run 5 random trials with adjust=False.
        for _ in range(5):
            self._run_trial(adjust_flag=False)

if __name__ == '__main__':
    unittest.main()


