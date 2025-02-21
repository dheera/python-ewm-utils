from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="ewm_utils.linear_regression",
        sources=["ewm_utils/linear_regression.pyx"],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        name="ewm_utils.quadratic_regression",
        sources=["ewm_utils/quadratic_regression.pyx"],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        name="ewm_utils.rolling_linear_regression",
        sources=["ewm_utils/rolling_linear_regression.pyx"],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        name="ewm_utils.rolling_quadratic_regression",
        sources=["ewm_utils/rolling_quadratic_regression.pyx"],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        name="ewm_utils.rolling_stats",
        sources=["ewm_utils/rolling_stats.pyx"],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        name="ewm_utils.rolling_correlation",
        sources=["ewm_utils/rolling_correlation.pyx"],
        include_dirs=[numpy.get_include()]
    ),
]

setup(
    name="ewm_utils",
    version="0.1",
    author="Dheera Venkatraman",
    author_email="dheera@dheera.net",
    description="Cython accelerated EWM regression and statistics",
    ext_modules=cythonize(extensions, annotate=True),
    packages=["ewm_utils"],
    include_dirs=[numpy.get_include()],
)
