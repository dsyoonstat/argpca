# examples/metrics.py
#
# Basic subspace metrics for ARG PCA examples.
#
# Currently provided:
#   - compute_principal_angles(U, V)

from __future__ import annotations

import numpy as np


def compute_principal_angles(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute principal angles (in radians) between two subspaces in R^p.

    This function assumes that **rows** of U and V form orthonormal bases of
    their respective subspaces:

        U : (k1, p),   U @ U.T ≈ I_{k1}
        V : (k2, p),   V @ V.T ≈ I_{k2}

    Under this convention, if we define A = U^T and B = V^T (column-orthonormal
    bases), then the singular values of A^T B equal those of U @ V.T. We use:

        M = U @ V.T   ∈ R^{k1×k2}
        σ_i = singular values of M
        θ_i = arccos(σ_i),

    where θ_i are the principal angles between the two subspaces.

    Parameters
    ----------
    U : (k1, p) array_like
        Row-orthonormal basis for the first subspace (each row is a unit
        vector in R^p, and rows are mutually orthogonal).
    V : (k2, p) array_like
        Row-orthonormal basis for the second subspace, same feature
        dimension p as U.

    Returns
    -------
    angles : (q,) ndarray
        Principal angles in radians, sorted in non-decreasing order:
            0 <= θ_1 <= θ_2 <= ... <= θ_q <= π/2,
        where q = min(k1, k2). The cosines of these angles are the singular
        values of U @ V.T.

    Raises
    ------
    ValueError
        If U or V are not 2D, or if their feature dimensions (number of
        columns) do not match.
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)

    if U.ndim != 2 or V.ndim != 2:
        raise ValueError("`U` and `V` must both be 2D arrays.")
    if U.shape[1] != V.shape[1]:
        raise ValueError(
            f"`U` and `V` must have the same number of columns (feature dimension p). "
            f"Got U.shape={U.shape}, V.shape={V.shape}."
        )

    # M = U V^T: its singular values are cosines of principal angles.
    M = U @ V.T  # shape (k1, k2)
    svals = np.linalg.svd(M, full_matrices=False, compute_uv=False)

    # Numerical guard: cos(theta) must lie in [0, 1].
    svals = np.clip(svals, 0.0, 1.0)

    # Singular values are returned in non-increasing order:
    #   1 >= σ_1 >= σ_2 >= ...
    # arccos is decreasing, so the resulting angles are in non-decreasing order:
    #   0 <= θ_1 <= θ_2 <= ...
    angles = np.arccos(svals)
    return angles