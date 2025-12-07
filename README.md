# argpca: Adaptive Reference-Guided PCA for High-Dimension, Low-Sample Size Data

`argpca` is a Python implementation of **Adaptive Reference-Guided PCA (ARG-PCA)**,  
a method designed for **high-dimension, low-sample-size (HDLSS)** data proposed in  
*[Yoon and Jung (2025)](<https://onlinelibrary.wiley.com/doi/full/10.1002/sta4.70081>)*.  
The method provides **improved asymptotic accuracy** for estimating the principal  
component (PC) subspace when informative **reference vectors** are available.

This package implements the full ARG estimator and its PCA counterpart, including:

- Efficient **Gram-matrix PCA** for HDLSS settings (p ≫ n)
- Computation of the **ARG PC subspace** (function API)
- A full **ARGPCA estimator** (scikit-learn–style)
- Tools for **benchmarking**, **simulations**, and **real-data analysis** (NASDAQ example)

---

## Why ARG-PCA?

Classical PCA performs poorly in HDLSS settings: the sample PC subspace is **inconsistent**,  
and the principal angles between the sample and true PC subspaces converge to a *non-zero*  
random limit. When **prior information** is available—such as domain-specific directions  
known (or believed) to be aligned with the true PC subspace—ARG-PCA leverages this to  
improve estimation accuracy. A representative example is the normalized vector of ones,  
often used in financial applications to reflect the common market factor,  
as in the capital asset pricing model. The ARG PC subspace estimator asymptotically  
outperforms the naive PCA based estimator. ARG-PCA is built based on ARG PC subspace estimator.

---

## Features

- **ARG PC subspace estimator** (`compute_arg_pc_subspace`)
- **ARG-PCA** (`ARGPCA`) with a scikit-learn–compatible API  
- **Fast PCA for HDLSS settings** via Gram-matrix eigen-decomposition  
- **Simulation and real-data examples** reproducing the empirical results in  
  *[Yoon and Jung (2025)](<https://onlinelibrary.wiley.com/doi/full/10.1002/sta4.70081>)*

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/dsyoonstat/argpca.git
cd argpca
pip install -e .
```

---

## Quick Start

### ARG-PCA Estimator

```python
import numpy as np
from argpca.pca import ARGPCA

# X: (n, p) data matrix
# V: (r, p) reference vectors (rows)
X = ...
V = ...

model = ARGPCA(n_components=2)
model.fit(X, reference_vectors=V)

scores     = model.transform(X)     # (n, 2)
components = model.components_      # (2, p)
```

### Direct subspace extraction

```python
from argpca.pca import compute_arg_pc_subspace

U_arg = compute_arg_pc_subspace(
    samples=X,
    reference_vectors=V,
    n_components=2,
)
```

---

## Public API

```python
from argpca import ARGPCA, compute_arg_pc_subspace

# advanced / low-level utilities
from argpca import utils
# or, more explicitly:
from argpca.utils import GramSpectrum, compute_gram_spectrum, recover_spike_directions
```

---

## Repository Structure

```
argpca/
 ├── src/argpca/
 │    ├── pca.py               # ARGPCA + subspace logic
 │    ├── utils.py             # GramSpectrum, Gram PCA utilities
 │    └── __init__.py
 │
 ├── examples/
 │    ├── simulation/          # Monte Carlo experiments
 │    │     ├── simulation.py
 │    │     ├── dgps.py
 │    │     └── metrics.py
 │    ├── realdata/            # NASDAQ 2024-12 analysis
 │    │     ├── realdata.py
 │    │     └── *.csv
 │
 ├── tests/                    # pytest unit tests
 └── pyproject.toml
```

---

## Running the Simulations

To reproduce the numerical experiments from 
*[Yoon and Jung (2025)](<https://onlinelibrary.wiley.com/doi/full/10.1002/sta4.70081>)*, run: 

```bash
python examples/simulation/simulation.py
```

Results are stored under:

```
examples/simulation/results/
```

---

## Real Data Example (NASDAQ 2024-12)

```bash
python examples/realdata/realdata.py
```

This script generates a scatter plot comparing:

- ARGPCA PC1–PC2 scores  
- Standard PCA PC1–PC2 scores  

Saved as:

```
examples/realdata/nasdaq_2024_12_pc_scores.png
```

For reference vectors, normalized vector of ones and 2024 mean log-returns were used.

---

## License

This project is released under the **MIT License**.

---

## Citation

If you use this package in academic work, please cite:

```bibtex
@article{Yoon2025,
  author    = {Yoon, Dongsun and Jung, Sungkyu},
  title     = {Adaptive Reference-Guided Estimation of Principal Component Subspace in High Dimensions},
  journal   = {Stat},
  volume    = {14},
  number    = {3},
  pages     = {e70081},
  year      = {2025},
  doi       = {10.1002/sta4.70081},
  url       = {https://onlinelibrary.wiley.com/doi/abs/10.1002/sta4.70081}
}
```

---

## Contributing

Pull requests and bug reports are welcome.  
Please use GitHub Issues for questions or feature requests.