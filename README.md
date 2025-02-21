# ewm_utils

**ewm_utils** is a high-performance Python package that implements exponentially weighted moving (EWM) statistics and regression methods using Cython. By leveraging Cython for computationally intensive routines, the package provides fast implementations for various EWM calculations, including:

- **Linear Regression** (with optional robust outlier rejection via RANSAC)
- **Quadratic Regression** (with robust outlier rejection)
- **Rolling EWM Linear Regression** (explicit and recursive update modes)
- **Rolling EWM Quadratic Regression**
- **Rolling EWM Statistics** (EWM mean and standard deviation)
- **Rolling EWM Correlation**

These modules are designed for efficient online (rolling) computations over streaming data or large datasets.

---

## Features

- **Cython-Accelerated Performance:**  
  All key routines are implemented in Cython

- **Flexible Modes:**  
  Support for explicit mode (storing all data with computed weights) and recursive mode (updating running statistics).

- **Robust Regression Options:**  
  Optional RANSAC-based outlier rejection for both linear and quadratic regressions.

- **Rolling Computations:**  
  Efficiently update statistics as new data arrives.

---

## Installation

Clone the repository and build the Cython extensions:

```bash
git clone <repository-url>
cd <repository-directory>
python setup.py build_ext --inplace
python setup.py install

