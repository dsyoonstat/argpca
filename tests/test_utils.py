# tests/test_utils.py
#
# Unit tests for argpca.utils:
#   - GramSpectrum and compute_gram_spectrum
#   - recover_spike_directions
#
# Run with:
#     pytest
# or:
#     pytest tests/test_utils.py

import numpy as np
import pytest

from argpca.utils import (
    GramSpectrum,
    compute_gram_spectrum,
    recover_spike_directions,
)


def test_compute_gram_spectrum_shapes_and_rank():
    """Check that compute_gram_spectrum returns correct shapes and metadata."""
    rng = np.random.default_rng(0)
    n, p = 20, 50  # HDLSS: p >> n
    X = rng.normal(size=(n, p))

    spectrum = compute_gram_spectrum(X)

    # Check core structure
    assert isinstance(spectrum, GramSpectrum)
    assert spectrum.Xc.shape == (n, p)
    assert spectrum.eigvals.shape == (n,)
    assert spectrum.eigvecs.shape == (n, n)
    assert spectrum.n_samples == n
    assert spectrum.n_features == p

    # Rank must be n - 1 for centered data in HDLSS
    assert spectrum.rank == n - 1

    # Eigenvalues must be sorted in descending order
    assert np.all(spectrum.eigvals[:-1] >= spectrum.eigvals[1:] - 1e-12)


def test_recover_spike_directions_matches_covariance_eig():
    """
    The spike eigenvalues/eigenvectors recovered via Gram-matrix PCA
    must match those of the full covariance matrix (up to numerical tolerance).
    """
    rng = np.random.default_rng(1)
    n, p = 30, 100
    X = rng.normal(size=(n, p))

    spectrum = compute_gram_spectrum(X)
    m = 5  # number of leading components

    U_spike, lam_spike, l_tilde = recover_spike_directions(spectrum, m)

    # Basic shape checks
    assert U_spike.shape == (p, m)
    assert lam_spike.shape == (m,)
    assert isinstance(l_tilde, float)

    # Full covariance-based PCA
    Xc = X - X.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / float(n)  # (p × p)

    w_full, V_full = np.linalg.eigh(S)
    idx = np.argsort(w_full)[::-1]  # descending order

    # Compare eigenvalues
    assert np.allclose(lam_spike, w_full[idx][:m], rtol=1e-5, atol=1e-7)

    # Compare subspaces (ignore sign/rotation)
    U = U_spike          # (p × m)
    V = V_full[:, idx[:m]]

    # Subspace distance: || UUᵀ - VVᵀ ||_F should be small
    P_U = U @ U.T
    P_V = V @ V.T
    subspace_diff = np.linalg.norm(P_U - P_V, ord="fro")
    assert subspace_diff < 1e-5


def test_recover_spike_directions_ltilde_as_bulk_mean():
    """Check that l_tilde equals the mean of non-spike eigenvalues."""
    rng = np.random.default_rng(2)
    n, p = 25, 80
    X = rng.normal(size=(n, p))

    spectrum = compute_gram_spectrum(X)
    k = spectrum.rank
    m = 3

    _, _, l_tilde = recover_spike_directions(spectrum, m)

    non_spike = spectrum.eigvals[m:k]
    expected = float(non_spike.mean()) if non_spike.size > 0 else 0.0

    assert np.allclose(l_tilde, expected, rtol=1e-6, atol=1e-8)


def test_recover_spike_directions_ltilde_zero_when_no_bulk():
    """
    When n_components equals the rank of the centered data,
    there is no non-spike eigenvalue and l_tilde must be 0.
    """
    rng = np.random.default_rng(3)
    n, p = 15, 40
    X = rng.normal(size=(n, p))

    spectrum = compute_gram_spectrum(X)
    k = spectrum.rank  # = n - 1

    _, _, l_tilde = recover_spike_directions(spectrum, n_components=k)
    assert l_tilde == pytest.approx(0.0, abs=1e-12)


def test_check_n_components_validation():
    """Check validation logic for spike number selection."""
    rng = np.random.default_rng(4)
    n, p = 10, 30
    X = rng.normal(size=(n, p))

    spectrum = compute_gram_spectrum(X)
    k = spectrum.rank

    # Valid values
    spectrum.check_n_components(1)
    spectrum.check_n_components(k)

    # Invalid: non-positive
    with pytest.raises(ValueError):
        spectrum.check_n_components(0)

    # Invalid: exceeds rank
    with pytest.raises(ValueError):
        spectrum.check_n_components(k + 1)

    # Invalid: non-integer
    with pytest.raises(TypeError):
        spectrum.check_n_components(1.5)