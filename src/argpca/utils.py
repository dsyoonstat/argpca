# utils.py
#
# Shared linear-algebra utilities for ARG PCA under HDLSS (p > n) settings.
#
# Responsibilities:
#   - Center the data and perform the Gram-matrix eigendecomposition exactly once.
#   - Package the result into a GramSpectrum object.
#   - Provide a convenient way to recover leading spike directions in feature
#     space and the bulk (non-spike) eigenvalue level.

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class GramSpectrum:
    """
    Container for the Gram-matrix eigendecomposition of centered data
    in (primarily) HDLSS (p > n) settings.

    Assumptions
    ----------
    - We typically work in a high-dimensional, low-sample-size regime with p > n.
    - Data are centered by subtracting the sample mean.
    - The Gram matrix is defined as
          G = (1/n) Xc Xc^T,
      where Xc is the centered data matrix.
    - The nonzero spectrum of G coincides (up to numerical error) with the
      nonzero spectrum of the sample covariance
          S = (1/n) Xc^T Xc.

    Attributes
    ----------
    Xc : (n, p) ndarray
        Centered data matrix (rows = samples, columns = features).
    eigvals : (n,) ndarray
        Eigenvalues of the Gram matrix G, sorted in descending order.
        The first `rank` entries correspond (up to numerical error) to the
        nonzero spectrum and coincide with the eigenvalues of the sample
        covariance S on its nonzero spectrum.
    eigvecs : (n, n) ndarray
        Corresponding orthonormal eigenvectors of G, with columns aligned
        with `eigvals`.
    n_samples : int
        Number of samples n.
    n_features : int
        Number of features p.
    rank : int
        Effective nonzero spectrum size. With centering, the rank of G is
        at most min(n - 1, p). We record
            rank = min(n_samples - 1, n_features),
        which equals n_samples - 1 in the intended HDLSS regime (p > n).
    """

    Xc: np.ndarray
    eigvals: np.ndarray
    eigvecs: np.ndarray
    n_samples: int
    n_features: int
    rank: int

    def check_n_components(self, n_components: int) -> None:
        """
        Validate that `n_components` lies in [1, rank].

        Parameters
        ----------
        n_components : int
            Number of leading components / spikes.

        Raises
        ------
        TypeError
            If `n_components` is not an integer.
        ValueError
            If `n_components` is out of the admissible range or if there
            are fewer than two samples.
        """
        if not isinstance(n_components, int):
            raise TypeError("`n_components` must be an integer.")
        if self.n_samples < 2:
            raise ValueError("Need at least two samples (n >= 2).")

        max_components = self.rank
        if not (1 <= n_components <= max_components):
            raise ValueError(
                f"`n_components` must be in [1, {max_components}], "
                f"got {n_components}."
            )


def compute_gram_spectrum(samples: np.ndarray) -> GramSpectrum:
    """
    Compute the Gram-matrix eigendecomposition for centered data.

    Given samples X with shape (n, p), this function:

    1. Centers X to obtain Xc.
    2. Forms the Gram matrix
           G = (1/n) Xc Xc^T.
    3. Computes its eigendecomposition.
    4. Sorts eigenvalues (and corresponding eigenvectors) in descending order.
    5. Sets
           rank = min(n - 1, p),
       reflecting the effective nonzero spectrum size under centering.

    Parameters
    ----------
    samples : (n, p) array_like
        Data matrix with rows = samples, columns = features. The intended
        use is HDLSS (p > n), but the function does not explicitly enforce
        this and remains well-defined more generally.

    Returns
    -------
    spectrum : GramSpectrum
        Container with centered data, Gram eigenpairs, and metadata.
    """
    X = np.asarray(samples)
    if X.ndim != 2:
        raise ValueError(f"`samples` must be 2D, got {X.ndim}D.")

    n, p = X.shape
    if n < 2:
        raise ValueError("Need at least two samples (n >= 2).")

    # Center data
    Xc = X - X.mean(axis=0, keepdims=True)  # (n, p)

    # Gram matrix and eigendecomposition
    G = (Xc @ Xc.T) / float(n)              # (n, n), symmetric PSD
    w, Q = np.linalg.eigh(G)                # w ascending
    idx_desc = np.argsort(w)[::-1]          # sort descending
    w = w[idx_desc]
    Q = Q[:, idx_desc]

    # Effective nonzero spectrum size under centering
    rank = min(n - 1, p)
    if rank < 0:
        rank = 0

    return GramSpectrum(
        Xc=Xc,
        eigvals=w,
        eigvecs=Q,
        n_samples=n,
        n_features=p,
        rank=rank,
    )


def recover_spike_directions(
    spectrum: GramSpectrum,
    n_components: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Recover top-`n_components` eigen-directions in feature space
    and the bulk (non-spike) eigenvalue level.

    This uses the standard dual relation between the Gram matrix
        G = (1/n) Xc Xc^T
    and the sample covariance
        S = (1/n) Xc^T Xc.

    Given Gram eigenpairs (w_i, q_i), the corresponding eigenvectors
    of S in feature space are
        u_i = Xc^T q_i / sqrt(n * w_i).

    Parameters
    ----------
    spectrum : GramSpectrum
        Precomputed Gram spectrum for centered data.
    n_components : int
        Number of leading spike directions (m). In the ARG PCA context,
        this is the number of spikes / leading components.

    Returns
    -------
    U_spike : (p, m) ndarray
        Feature-space eigenvectors associated with the top-m eigenvalues
        of S. Columns are (approximately) orthonormal directions in R^p.
    lam_spike : (m,) ndarray
        Corresponding eigenvalues (lambda_1, ..., lambda_m), i.e., the
        top-m eigenvalues of S = (1/n) Xc^T Xc (coinciding with those of G
        on the nonzero spectrum).
    l_tilde : float
        Mean of the non-spike eigenvalues among the nonzero spectrum:
            l_tilde = mean(lambda_{m+1}, ..., lambda_rank),
        or 0.0 if rank <= m.

    Notes
    -----
    - Only the first `rank` entries of `spectrum.eigvals` correspond to the
      effective nonzero spectrum. We define the bulk level using
      eigvals[m:rank].
    """
    spectrum.check_n_components(n_components)

    Xc = spectrum.Xc
    w = spectrum.eigvals
    Q = spectrum.eigvecs
    n = spectrum.n_samples
    p = spectrum.n_features
    k = spectrum.rank
    m = n_components

    # Top-m eigenpairs of the Gram / covariance
    lam_spike = w[:m].copy()      # (m,)
    Q_spike = Q[:, :m]            # (n, m)

    # Recover feature-space eigenvectors of S
    denom = np.sqrt(np.maximum(lam_spike * float(n), 1e-32))
    U_spike = (Xc.T @ Q_spike) / denom[None, :]  # (p, m)

    # Bulk level: mean of non-spike eigenvalues among the nonzero spectrum
    if k > m:
        l_tilde = float(np.mean(w[m:k]))
    else:
        l_tilde = 0.0

    return U_spike, lam_spike, l_tilde