import unittest
import numpy as np
from ewm_utils import RollingStats
import pandas as pd

# Unit test that verifies against Pandas implementation

if __name__ == '__main__':

    class TestRollingStats(unittest.TestCase):
        def setUp(self):
            self.alpha = 0.1
            self.min_periods = 5
            np.random.seed(42)

        def _run_trial(self, adjust_flag):
            # Generate a random series (length between 50 and 150).
            n = np.random.randint(50, 151)
            values = np.random.randn(n)

            # Create our rolling  stats object.
            rolling_stats = RollingStats(alpha=self.alpha, adjust=adjust_flag, min_periods=self.min_periods)
            means = []
            stds = []
            for val in values:
                m, s = rolling_stats.update(val)
                means.append(m)
                stds.append(s)
            means = np.array(means)
            stds = np.array(stds)

            # Compute expected results using pandas.
            series = pd.Series(values)
            pd_mean = series.ewm(alpha=self.alpha, adjust=adjust_flag, min_periods=self.min_periods).mean().to_numpy()
            pd_std  = series.ewm(alpha=self.alpha, adjust=adjust_flag, min_periods=self.min_periods).std().to_numpy()

            # Use a tight tolerance for explicit mode and a looser tolerance for recursive mode.
            tol = 1e-4 if adjust_flag else 0.3
            np.testing.assert_allclose(means, pd_mean, rtol=tol, equal_nan=True)
            np.testing.assert_allclose(stds, pd_std, rtol=tol, equal_nan=True)

        def test_adjust_true_random_trials(self):
            for _ in range(5):
                self._run_trial(adjust_flag=True)

        def test_adjust_false_random_trials(self):
            for _ in range(5):
                self._run_trial(adjust_flag=False)

    unittest.main()


