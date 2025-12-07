# tests/test_pca.py
#
# Unit tests for argpca.pca:
#   - _orthonormalize_reference_rows
#   - compute_arg_pc_subspace
#   - ARGPCA
#
# Run with:
#     pytest
# or:
#     pytest tests/test_pca.py

import numpy as np
import pytest

from argpca.pca import (
    _orthonormalize_reference_rows,
    compute_arg_pc_subspace,
    ARGPCA,
)
from argpca.utils import compute_gram_spectrum


# ---------------------------------------------------------------------------
# Tests for _orthonormalize_reference_rows
# ---------------------------------------------------------------------------

def test_orthonormalize_reference_rows_row_orthonormal():
    """Rows of the returned matrix should be orthonormal."""
    rng = np.random.default_rng(0)
    r, p = 5, 40
    V = rng.normal(size=(r, p))

    V_orth = _orthonormalize_reference_rows(V, p=p)  # (r_eff, p)

    # Row-orthonormality: V_orth V_orth^T = I_{r_eff}
    prod = V_orth @ V_orth.T
    r_eff = V_orth.shape[0]
    assert V_orth.shape == (r_eff, p)
    assert np.allclose(prod, np.eye(r_eff), atol=1e-6)


def test_orthonormalize_reference_rows_preserves_span():
    """The row span of the orthonormal basis should match the original span."""
    rng = np.random.default_rng(1)
    r, p = 4, 30
    V = rng.normal(size=(r, p))

    V_orth = _orthonormalize_reference_rows(V, p=p)  # (r_eff, p)

    # Compare projections onto row spans using a few random test vectors.
    # Original projection: P_V(x) = V^T (VV^T)^+ V x
    # Orthonormal projection: P_Vorth(x) = V_orth^T V_orth x
    VVt = V @ V.T
    # Use pseudoinverse for numerical stability
    VVt_pinv = np.linalg.pinv(VVt)

    for _ in range(5):
        x = rng.normal(size=(p,))
        proj_orig = V.T @ (VVt_pinv @ (V @ x))
        proj_orth = V_orth.T @ (V_orth @ x)
        assert np.allclose(proj_orig, proj_orth, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests for compute_arg_pc_subspace
# ---------------------------------------------------------------------------

def test_compute_arg_pc_subspace_basic_shape_and_orthonormality():
    """ARG PC subspace should have correct shape and (optionally) orthonormal rows."""
    rng = np.random.default_rng(2)
    n, p = 20, 60
    X = rng.normal(size=(n, p))
    V = rng.normal(size=(3, p))

    m = 4
    components = compute_arg_pc_subspace(
        samples=X,
        reference_vectors=V,
        n_components=m,
        orthonormal=True,
    )

    # Shape: (m, p), sklearn-like convention
    assert components.shape == (m, p)

    # Rows should be approximately orthonormal
    gram = components @ components.T
    assert np.allclose(gram, np.eye(m), atol=1e-6)


def test_compute_arg_pc_subspace_with_precomputed_spectrum_matches_direct():
    """Using a precomputed GramSpectrum should give the same result as computing inside."""
    rng = np.random.default_rng(3)
    n, p = 25, 80
    X = rng.normal(size=(n, p))
    V = rng.normal(size=(4, p))
    m = 3

    # Direct call (spectrum computed inside)
    components_direct = compute_arg_pc_subspace(
        samples=X,
        reference_vectors=V,
        n_components=m,
        orthonormal=True,
        spectrum=None,
    )

    # Call with precomputed spectrum
    spectrum = compute_gram_spectrum(X)
    components_reuse = compute_arg_pc_subspace(
        samples=X,
        reference_vectors=V,
        n_components=m,
        orthonormal=True,
        spectrum=spectrum,
    )

    # They should match up to numerical precision
    assert components_direct.shape == components_reuse.shape
    assert np.allclose(components_direct, components_reuse, atol=1e-8)


# ---------------------------------------------------------------------------
# Tests for ARGPCA estimator
# ---------------------------------------------------------------------------

def test_argpca_fit_sets_core_attributes():
    """ARGPCA.fit should populate core attributes with consistent shapes."""
    rng = np.random.default_rng(4)
    n, p = 30, 100
    X = rng.normal(size=(n, p))
    V = rng.normal(size=(5, p))

    m = 4
    model = ARGPCA(n_components=m).fit(X, V)

    # Basic metadata
    assert model.n_samples_ == n
    assert model.n_features_ == p
    assert model.n_components_ == m

    # Mean and components
    assert model.mean_.shape == (p,)
    assert model.components_.shape == (m, p)

    # Variance-related attributes
    assert model.explained_variance_.shape == (m,)
    assert model.explained_variance_ratio_.shape == (m,)
    assert model.noise_variance_ >= 0.0


def test_argpca_transform_and_inverse_transform_shapes():
    """transform and inverse_transform should produce consistent shapes."""
    rng = np.random.default_rng(5)
    n, p = 40, 120
    X = rng.normal(size=(n, p))
    V = rng.normal(size=(6, p))

    m = 5
    model = ARGPCA(n_components=m).fit(X, V)

    Z = model.transform(X)
    assert Z.shape == (n, m)

    X_rec = model.inverse_transform(Z)
    assert X_rec.shape == (n, p)


def test_argpca_inverse_transform_recovers_mean():
    """Reconstruction should preserve the original data mean (since centering is used)."""
    rng = np.random.default_rng(6)
    n, p = 25, 50
    X = rng.normal(size=(n, p))
    V = rng.normal(size=(4, p))

    m = 3
    model = ARGPCA(n_components=m).fit(X, V)

    Z = model.transform(X)
    X_rec = model.inverse_transform(Z)

    # Means should be very close
    mean_orig = X.mean(axis=0)
    mean_rec = X_rec.mean(axis=0)
    assert np.allclose(mean_rec, mean_orig, atol=1e-6)


def test_argpca_fit_transform_consistency():
    """fit_transform should be equivalent to fit followed by transform."""
    rng = np.random.default_rng(7)
    n, p = 20, 40
    X = rng.normal(size=(n, p))
    V = rng.normal(size=(3, p))

    m = 2
    model = ARGPCA(n_components=m)

    Z1 = model.fit_transform(X, V)
    Z2 = model.transform(X)

    assert Z1.shape == (n, m)
    assert np.allclose(Z1, Z2, atol=1e-10)


def test_argpca_get_covariance_basic_properties():
    """get_covariance should return a symmetric (p, p) matrix."""
    rng = np.random.default_rng(8)
    n, p = 15, 60
    X = rng.normal(size=(n, p))
    V = rng.normal(size=(4, p))

    m = 3
    model = ARGPCA(n_components=m).fit(X, V)

    cov = model.get_covariance()
    assert cov.shape == (p, p)

    # Symmetry check
    assert np.allclose(cov, cov.T, atol=1e-8)

    
def test_argpca_scores_covariance_matches_explained_variance():
    """
    In PCA, the covariance of the scores should be diagonal with entries
    equal to the explained variances. ARG PCA should respect the same
    property for the learned components.
    """
    rng = np.random.default_rng(9)
    n, p = 50, 200
    X = rng.normal(size=(n, p))
    V = rng.normal(size=(5, p))

    m = 4
    model = ARGPCA(n_components=m).fit(X, V)

    # Scores in the learned ARG PCA directions
    Z = model.transform(X)  # (n, m)
    # Sample covariance of the scores
    S_scores = (Z.T @ Z) / float(n)  # (m, m)

    # Explained variances should appear on the diagonal, off-diagonals ~ 0
    diag_S = np.diag(S_scores)
    assert np.allclose(diag_S, model.explained_variance_, atol=1e-6)

    off_diag = S_scores - np.diag(diag_S)
    assert np.allclose(off_diag, 0.0, atol=1e-6)


def test_argpca_projected_subspace_pca_eigenvalues():
    """
    ARGPCA is conceptually:
      1) project onto the ARG subspace,
      2) perform PCA in that subspace.

    We check that the eigenvalues of the covariance in the ARG subspace
    match the stored explained_variance_.
    """
    rng = np.random.default_rng(10)
    n, p = 40, 150
    X = rng.normal(size=(n, p))
    V = rng.normal(size=(6, p))

    m = 3
    model = ARGPCA(n_components=m).fit(X, V)

    # ARG basis (row-wise) learned in fit
    B = model._arg_basis_.T  # (p, m), columns = ARG basis
    # Centered data using the same mean as in the model
    Xc = X - model.mean_
    Z = Xc @ B  # (n, m) projected onto ARG subspace

    # Covariance in ARG coordinates
    S_H = (Z.T @ Z) / float(n)  # (m, m)
    eigvals, _ = np.linalg.eigh(S_H)
    eigvals = eigvals[::-1]  # descending

    assert eigvals.shape == model.explained_variance_.shape
    assert np.allclose(eigvals, model.explained_variance_, atol=1e-6)


def test_argpca_invalid_n_components_raises():
    """
    If n_components exceeds the effective rank (≈ n_samples - 1 under
    centering), ARGPCA.fit should raise a ValueError.
    """
    rng = np.random.default_rng(11)
    n, p = 10, 100
    X = rng.normal(size=(n, p))
    V = rng.normal(size=(3, p))

    # rank = min(n - 1, p) = 9, so n_components = 10 is invalid
    with pytest.raises(ValueError):
        ARGPCA(n_components=n).fit(X, V)