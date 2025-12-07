# examples/dgps.py
#
# Data-generating processes (DGPs) and sampling utilities for ARG PCA examples.
#
# Provided functionality:
#   - generate_basis(p)
#   - sigma_single_spike(p, coefs)
#   - sigma_multi_spike(p, coefs)
#   - sample_normal(Sigma, n, mu, seed)
#   - sample_t(Sigma, nu, n, mu, seed)
#   - generate_reference_vectors(E, A)
#
# Conventions
# ----------
# - p is always the feature dimension (row length).
# - Basis vectors and reference vectors are stored as rows of length p:
#       generate_basis(p)      -> (4, p)  rows = e1, e2, e3, e4
#       generate_reference...  -> (r, p)  rows = reference vectors
# - Samples are stored in shape (n, p): rows = observations.

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Basis vectors generator
# ---------------------------------------------------------------------------

def generate_basis(p: int) -> np.ndarray:
    """
    Generate a fixed orthonormal system (e1, e2, e3, e4) in R^p.

    We assume p is divisible by 4 and construct:

        e1 = (1/sqrt(p)) * [ + + + + ]
        e2 = (1/sqrt(p)) * [ + + - - ]
        e3 = (1/sqrt(p)) * [ + - - + ]
        e4 = (1/sqrt(p)) * [ + - + - ],

    where '+' and '-' blocks each have length p/4.

    Parameters
    ----------
    p : int
        Feature dimension. Must be divisible by 4.

    Returns
    -------
    E : (4, p) ndarray
        Matrix whose rows are e1, e2, e3, e4 (each of length p).

    Raises
    ------
    ValueError
        If p is not divisible by 4.
    """
    if p <= 0:
        raise ValueError("`p` must be positive.")
    if p % 4 != 0:
        raise ValueError("`p` must be divisible by 4.")

    q = p // 4
    s = 1.0 / np.sqrt(p)

    ones = np.ones
    e1 = s * np.concatenate([ones(q),  ones(q),  ones(q),  ones(q)])
    e2 = s * np.concatenate([ones(q),  ones(q), -ones(q), -ones(q)])
    e3 = s * np.concatenate([ones(q), -ones(q), -ones(q),  ones(q)])
    e4 = s * np.concatenate([ones(q), -ones(q),  ones(q), -ones(q)])

    # Rows = e1, e2, e3, e4
    E = np.vstack([e1, e2, e3, e4])  # (4, p)
    return E


# ---------------------------------------------------------------------------
# Covariance builders (spiked models)
# ---------------------------------------------------------------------------

def sigma_single_spike(
    p: int,
    coefs: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a single-spike covariance matrix:

        Sigma = c1 * p * e1 e1^T + c2 * I_p,

    where e1 is the first basis vector from `generate_basis(p)`.

    Parameters
    ----------
    p : int
        Feature dimension.
    coefs : (float, float)
        Tuple (c1, c2). Here:
            c1 : spike strength multiplier
            c2 : noise variance (isotropic)

    Returns
    -------
    Sigma : (p, p) ndarray
        Covariance matrix with a single spike in direction e1.
    e1_row : (1, p) ndarray
        Principal spike direction, stored as a single **row vector**.

    Raises
    ------
    ValueError
        If p is not divisible by 4 (see `generate_basis`).
    """
    E = generate_basis(p)       # (4, p)
    e1 = E[0, :]                # (p,)
    c1, c2 = coefs

    Sigma = c1 * p * np.outer(e1, e1) + c2 * np.eye(p)
    e1_row = e1[None, :]        # (1, p) row vector
    return Sigma, e1_row


def sigma_multi_spike(
    p: int,
    coefs: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a two-spike covariance matrix:

        Sigma = c1 * p * e1 e1^T + c2 * p * e2 e2^T + c3 * I_p,

    where e1, e2 are the first two basis vectors from `generate_basis(p)`.

    Parameters
    ----------
    p : int
        Feature dimension.
    coefs : (float, float, float)
        Tuple (c1, c2, c3). Here:
            c1 : first spike strength
            c2 : second spike strength
            c3 : noise variance (isotropic)

    Returns
    -------
    Sigma : (p, p) ndarray
        Covariance matrix with two spike directions in span{e1, e2}.
    U_m : (2, p) ndarray
        True spike subspace basis. Each row is a spike direction (e1, e2).

    Raises
    ------
    ValueError
        If p is not divisible by 4 (see `generate_basis`).
    """
    E = generate_basis(p)       # (4, p)
    e1 = E[0, :]                # (p,)
    e2 = E[1, :]                # (p,)
    c1, c2, c3 = coefs

    Sigma = (
        c1 * p * np.outer(e1, e1)
        + c2 * p * np.outer(e2, e2)
        + c3 * np.eye(p)
    )
    U_m = np.vstack([e1, e2])   # (2, p) rows = spike directions
    return Sigma, U_m


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

def _validate_mu(mu: Optional[np.ndarray], p: int) -> np.ndarray:
    """Internal helper: ensure mean vector has length p."""
    if mu is None:
        return np.zeros(p, dtype=float)
    mu_arr = np.asarray(mu, dtype=float).reshape(-1)
    if mu_arr.shape[0] != p:
        raise ValueError(f"`mu` must have length {p}, got {mu_arr.shape[0]}.")
    return mu_arr


def sample_normal(
    Sigma: np.ndarray,
    n: int,
    mu: Optional[np.ndarray] = None,
    seed: Optional[int] = 725,
) -> np.ndarray:
    """
    Draw samples from a multivariate normal distribution:

        X_i ~ N(mu, Sigma),  i = 1, ..., n.

    Parameters
    ----------
    Sigma : (p, p) array_like
        Symmetric positive-definite covariance matrix.
    n : int
        Number of samples to draw.
    mu : (p,) array_like, optional
        Mean vector. If None, uses the zero vector in R^p.
    seed : int, optional
        Seed for NumPy's default_rng. If None, uses an unseeded generator.

    Returns
    -------
    X : (n, p) ndarray
        Generated samples, one per row.
    """
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("`Sigma` must be a square (p, p) matrix.")

    p = Sigma.shape[0]
    mu_vec = _validate_mu(mu, p)

    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(Sigma)          # (p, p)
    Z = rng.normal(size=(n, p))            # (n, p)
    X = mu_vec[None, :] + Z @ L.T          # (n, p)
    return X


def sample_t(
    Sigma: np.ndarray,
    nu: int,
    n: int,
    mu: Optional[np.ndarray] = None,
    seed: Optional[int] = 725,
) -> np.ndarray:
    """
    Draw samples from a multivariate t distribution with covariance `Sigma`.

    We construct X such that:
      - X ~ t_nu(mu, S) in the standard construction, and
      - Cov(X) = Sigma (for nu > 2).

    More precisely, if:

        Z ~ N(0, S),
        w ~ chi^2_nu,

    independent, then:

        T = Z / sqrt(w / nu)

    follows a multivariate t distribution with df = nu and scale S.
    We choose S so that Cov[T] = Sigma, i.e.:

        S = ((nu - 2) / nu) * Sigma.

    This matches the behaviour of MATLAB's:
        mvtrnd((nu-2)/nu * Sigma, nu, n)

    Parameters
    ----------
    Sigma : (p, p) array_like
        Target covariance matrix (must be symmetric positive-definite).
    nu : int
        Degrees of freedom. Must satisfy nu > 2 for finite covariance.
    n : int
        Number of samples to draw.
    mu : (p,) array_like, optional
        Mean vector. If None, uses the zero vector in R^p.
    seed : int, optional
        Seed for NumPy's default_rng. If None, uses an unseeded generator.

    Returns
    -------
    X : (n, p) ndarray
        Generated samples, one per row.

    Raises
    ------
    ValueError
        If nu <= 2 or Sigma is not square.
    """
    if nu <= 2:
        raise ValueError("`nu` must be > 2 for finite covariance.")

    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("`Sigma` must be a square (p, p) matrix.")

    p = Sigma.shape[0]
    mu_vec = _validate_mu(mu, p)

    rng = np.random.default_rng(seed)

    # Choose S so that Cov[T] = Sigma
    S = ((nu - 2) / nu) * Sigma
    L = np.linalg.cholesky(S)             # (p, p)

    Z = rng.normal(size=(n, p))           # (n, p)
    w = rng.chisquare(df=nu, size=n)      # (n,)

    T = (Z @ L.T) / np.sqrt(w / nu)[:, None]  # (n, p), row-wise scaling
    X = mu_vec[None, :] + T
    return X


# ---------------------------------------------------------------------------
# Reference vector generator
# ---------------------------------------------------------------------------

def generate_reference_vectors(E: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Construct reference vectors as linear combinations of the basis e1..e4.

    Parameters
    ----------
    E : (4, p) array_like
        Basis matrix from `generate_basis(p)`. Each row is e_i in R^p.
    A : (r, 4) array_like
        Weight matrix. Each row a_j ∈ R^4 defines a reference vector:
            v_j = sum_{i=1}^4 a_{j,i} e_i.

    Returns
    -------
    V : (r, p) ndarray
        Reference vectors as rows in R^p. Row j is v_j.

    Notes
    -----
    Implemented as matrix multiplication:

        V = A @ E

    since A ∈ R^{r×4}, E ∈ R^{4×p}, hence V ∈ R^{r×p}.
    """
    E = np.asarray(E, dtype=float)
    A = np.asarray(A, dtype=float)

    if E.ndim != 2 or E.shape[0] != 4:
        raise ValueError("`E` must have shape (4, p) as returned by `generate_basis`.")
    if A.ndim != 2 or A.shape[1] != 4:
        raise ValueError("`A` must have shape (r, 4).")

    V = A @ E  # (r, p)
    return V