"""
argpca
======

Adaptive Reference Guided PCA (ARGPCA) for high-dimensional, low-sample-size (HDLSS)
settings.

This package provides:

Core Public API
---------------
- ``ARGPCA``:
      A scikit-learn–style estimator implementing ARG-guided PCA.
- ``compute_arg_pc_subspace``:
      Functional API returning the ARG PC subspace basis.

Advanced / Low-level Utilities
------------------------------
The module ``argpca.utils`` exposes additional Gram-matrix tools:

    - ``GramSpectrum``
    - ``compute_gram_spectrum``
    - ``recover_spike_directions``

These utilities are intentionally *not* re-exported at the top level, to keep
the public API minimal and stable. Advanced users may import them explicitly as:

    >>> from argpca.utils import compute_gram_spectrum

"""

from __future__ import annotations

# Core public API
from .pca import ARGPCA, compute_arg_pc_subspace

# Expose the utils module itself (not individual symbols)
from . import utils as utils

__all__ = [
    "ARGPCA",
    "compute_arg_pc_subspace",
    "utils",   # Advanced utilities module (not individually re-exported)
]