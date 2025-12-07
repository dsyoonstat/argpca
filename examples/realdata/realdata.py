# examples/realdata.py
#
# Real-data example for ARGPCA + baseline PCA comparison.
#
# This script:
#   1. Loads NASDAQ 2024-12 log-returns and 2024 mean log-return
#      vector from CSV files.
#   2. Constructs two reference vectors in R^p:
#        v₁ = (1, 1, ..., 1) / sqrt(p)
#        v₂ = 2024 mean log-return vector.
#   3. Fits ARGPCA with 2 components using these reference vectors.
#   4. Fits baseline PCA with 2 components using the same Gram-matrix
#      machinery as ARGPCA (via argpca.utils).
#   5. Optionally aligns the *sign* of ARGPCA scores to match baseline PCA
#      (component-wise) for reproducible plots.
#   6. Plots the first two ARG PC scores and the first two PCA scores together.
#
# Usage (from repo root):
#   python examples/realdata.py
#
# The script is intended as an example / reproducibility tool and is
# not part of the public argpca API.

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from argpca.pca import ARGPCA
from argpca.utils import compute_gram_spectrum, recover_spike_directions


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# CSV with log-returns: rows = dates, columns = assets
DEFAULT_LOGRETURNS_CSV = BASE_DIR / "nasdaq_2024_12_log_returns.csv"

# CSV with mean log-return vector: single row, same numeric columns as above
DEFAULT_MEANRETURNS_CSV = BASE_DIR / "nasdaq_2024_mean_log_returns.csv"

# Output figure
DEFAULT_FIG_PATH = BASE_DIR / "nasdaq_2024_12_pc_scores.png"


# ---------------------------------------------------------------------------
# Data loading + reference construction
# ---------------------------------------------------------------------------

def load_log_returns_and_references(
    log_returns_path: Path | str | None = None,
    mean_returns_path: Path | str | None = None,
) -> tuple[np.ndarray, List[str], np.ndarray]:
    """
    Load NASDAQ log-returns and construct two reference vectors.

    References
    ----------
    We build two reference vectors in R^p:
      v₁ = (1, 1, ..., 1) / sqrt(p)
      v₂ = 2024 mean log-return vector (aligned with the log-return columns).

    Parameters
    ----------
    log_returns_path : path-like, optional
        CSV file with log-returns. If None, uses DEFAULT_LOGRETURNS_CSV.
        The file is expected to have numeric columns for assets (and
        optionally non-numeric columns such as dates, which are ignored).
    mean_returns_path : path-like, optional
        CSV file with a single row of mean log-returns. If None, uses
        DEFAULT_MEANRETURNS_CSV. The numeric columns should either:
          - exactly match the asset columns of the log-returns CSV, or
          - be a single numeric row of length p.

    Returns
    -------
    X : (n, p) ndarray
        Log-return matrix, with rows corresponding to observations (e.g., days)
        and columns to assets.
    feature_names : list of str
        Names of the asset columns used in X.
    reference_vectors : (2, p) ndarray
        Two reference vectors stored as rows:
          reference_vectors[0, :] = v₁ = ones / sqrt(p)
          reference_vectors[1, :] = v₂ = mean log-return vector

    Raises
    ------
    ValueError
        If the mean-returns CSV cannot be aligned with the log-return columns.
    """
    if log_returns_path is None:
        log_returns_path = DEFAULT_LOGRETURNS_CSV
    if mean_returns_path is None:
        mean_returns_path = DEFAULT_MEANRETURNS_CSV

    log_returns_path = Path(log_returns_path)
    mean_returns_path = Path(mean_returns_path)

    # --- Load log-returns CSV ---
    df_log = pd.read_csv(log_returns_path)

    # Keep only numeric columns (e.g. drop date column)
    df_numeric = df_log.select_dtypes(include=[np.number])
    if df_numeric.empty:
        raise ValueError(
            f"No numeric columns found in log-returns file: {log_returns_path}"
        )

    X = df_numeric.to_numpy(dtype=float)
    feature_names: List[str] = list(df_numeric.columns)
    n, p = X.shape

    # --- Load mean log-returns CSV ---
    df_mean = pd.read_csv(mean_returns_path)
    df_mean_numeric = df_mean.select_dtypes(include=[np.number])
    if df_mean_numeric.empty:
        raise ValueError(
            f"No numeric columns found in mean-returns file: {mean_returns_path}"
        )

    # Try to align by column names first
    if set(feature_names).issubset(set(df_mean_numeric.columns)):
        # Use exactly the same columns, in the same order
        mean_vec = df_mean_numeric.loc[0, feature_names].to_numpy(dtype=float)
    elif df_mean_numeric.shape == (1, p):
        # Fallback: just assume order matches if lengths coincide
        mean_vec = df_mean_numeric.iloc[0].to_numpy(dtype=float)
    else:
        raise ValueError(
            "Mean log-returns CSV is not compatible with log-returns columns. "
            "Check that it has either matching column names or a single row "
            "with the same number of numeric columns."
        )

    # --- Construct reference vectors as rows ---
    one_row = np.ones(p, dtype=float)[None, :] / np.sqrt(p)  # (1, p)
    mean_row = mean_vec.reshape(1, -1)                       # (1, p)

    reference_vectors = np.vstack([one_row, mean_row])       # (2, p)

    return X, feature_names, reference_vectors


# ---------------------------------------------------------------------------
# Baseline PCA via GramSpectrum (no sklearn)
# ---------------------------------------------------------------------------

def compute_pca_scores_from_gram(
    samples: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """
    Compute baseline PCA scores using argpca.utils (Gram-matrix PCA).

    This uses the same centered data and Gram-matrix eigendecomposition
    that underlies ARGPCA, ensuring that the comparison between PCA and
    ARGPCA is internally consistent.

    Parameters
    ----------
    samples : (n, p) array_like
        Data matrix with rows = samples and columns = features.
    n_components : int
        Number of leading principal components to compute.

    Returns
    -------
    scores : (n, n_components) ndarray
        PCA scores, i.e., coordinates of each sample in the top
        principal component directions.

    Notes
    -----
    Let Xc be the centered data in R^{n×p}, and let U_spike ∈ R^{p×m}
    be the leading m eigenvectors of the sample covariance matrix
    S = (1/n) Xc^T Xc (as returned by ``recover_spike_directions``).
    Then the PCA score matrix is given by

        scores = Xc U_spike,

    where scores[i, j] is the score of sample i along component j.
    """
    X = np.asarray(samples)
    if X.ndim != 2:
        raise ValueError(f"`samples` must be 2D, got {X.ndim}D.")

    n, p = X.shape
    if n < 2:
        raise ValueError("Need at least two samples (n >= 2).")
    if n_components < 1:
        raise ValueError("`n_components` must be at least 1.")

    # Compute Gram spectrum once
    spectrum = compute_gram_spectrum(X)

    # Recover top principal directions (feature space)
    U_spike, _, _ = recover_spike_directions(
        spectrum,
        n_components=n_components,
    )  # U_spike: (p, m)

    # PCA scores = centered data × eigenvectors
    Xc = spectrum.Xc                       # (n, p)
    scores = Xc @ U_spike                  # (n, m)
    return scores


# ---------------------------------------------------------------------------
# Sign alignment helper (example-only)
# ---------------------------------------------------------------------------

def _align_argpca_sign_with_pca(
    arg_scores: np.ndarray,
    pca_scores: np.ndarray,
) -> np.ndarray:
    """
    Align the sign of ARGPCA scores to match baseline PCA scores.

    For each component j, we compute the inner product

        s_j = <ARG_j, PCA_j> = sum_i arg_scores[i, j] * pca_scores[i, j],

    and if s_j < 0, we flip the sign of the j-th ARGPCA score column.
    This preserves all geometry (subspaces are unchanged) but produces
    a plot with a consistent orientation, which is useful for matching
    published figures.

    Parameters
    ----------
    arg_scores : (n, m) ndarray
        ARGPCA scores (e.g., from ``ARGPCA.transform``).
    pca_scores : (n, m) ndarray
        Baseline PCA scores computed from the same data.

    Returns
    -------
    arg_scores_aligned : (n, m) ndarray
        Copy of ``arg_scores`` with columns possibly sign-flipped in order
        to align with the corresponding PCA components.

    Raises
    ------
    ValueError
        If the shapes of ``arg_scores`` and ``pca_scores`` do not match.
    """
    A = np.asarray(arg_scores, dtype=float)
    P = np.asarray(pca_scores, dtype=float)

    if A.shape != P.shape:
        raise ValueError(
            "`arg_scores` and `pca_scores` must have the same shape; "
            f"got {A.shape} vs {P.shape}."
        )

    n, m = A.shape
    if n == 0 or m == 0:
        return A.copy()

    A_aligned = A.copy()
    for j in range(m):
        s = float(np.dot(A_aligned[:, j], P[:, j]))
        if s < 0.0:
            A_aligned[:, j] *= -1.0

    return A_aligned


# ---------------------------------------------------------------------------
# ARGPCA + PCA comparison plot
# ---------------------------------------------------------------------------

def run_argpca_and_plot(
    n_components: int = 2,
    log_returns_path: Path | str | None = None,
    mean_returns_path: Path | str | None = None,
    fig_path: Path | str | None = None,
    align_signs: bool = True,
) -> None:
    """
    Fit ARGPCA and baseline PCA on NASDAQ log-returns and plot 2D scores.

    Workflow
    --------
    1. Load log-returns X and construct two reference vectors (ones, mean).
    2. Fit ARGPCA with ``n_components`` on X.
    3. Compute PCA scores with ``n_components`` using Gram-matrix PCA.
    4. Optionally align the *sign* of each ARGPCA component to match
       the corresponding PCA component (for plot orientation).
    5. Plot:
         - ARGPCA PC1 vs PC2 scores (blue dots),
         - PCA PC1 vs PC2 scores (red dots),
       in a shared 2D scatter plot.

    Parameters
    ----------
    n_components : int, default=2
        Number of components for ARGPCA and baseline PCA. We only plot
        the first two components in the figure.
    log_returns_path : path-like, optional
        CSV file for NASDAQ log-returns. If None, uses DEFAULT_LOGRETURNS_CSV.
    mean_returns_path : path-like, optional
        CSV file for mean log-returns. If None, uses DEFAULT_MEANRETURNS_CSV.
    fig_path : path-like, optional
        Output path for the PNG figure. If None, uses DEFAULT_FIG_PATH.
    align_signs : bool, default=True
        If True, flip the signs of ARGPCA score columns so that each
        component is aligned (by inner product) with the corresponding
        PCA component. This does not change the subspace, only the
        global sign convention for visualization / reproducibility.
    """
    # 1) Load data and references
    X, feature_names, reference_vectors = load_log_returns_and_references(
        log_returns_path=log_returns_path,
        mean_returns_path=mean_returns_path,
    )
    n, p = X.shape
    print(f"Loaded NASDAQ log-returns: n={n}, p={p}")
    print(f"Using {len(feature_names)} assets as features.")

    if n_components < 2:
        raise ValueError("`n_components` must be at least 2 to make a 2D plot.")

    # 2) Fit ARGPCA and get scores
    argpca = ARGPCA(n_components=n_components)
    argpca.fit(X, reference_vectors=reference_vectors)
    arg_scores = argpca.transform(X)  # (n, m)

    # 3) Baseline PCA scores via Gram-spectrum
    pca_scores = compute_pca_scores_from_gram(
        samples=X,
        n_components=n_components,
    )  # (n, m)

    # 4) Optionally align ARGPCA signs with PCA
    if align_signs:
        arg_scores = _align_argpca_sign_with_pca(arg_scores, pca_scores)
        print("Aligned ARGPCA score signs to match baseline PCA (component-wise).")

    # Extract first two dimensions
    arg_x, arg_y = arg_scores[:, 0], arg_scores[:, 1]
    pca_x, pca_y = pca_scores[:, 0], pca_scores[:, 1]

    # 5) Plot ARGPCA vs PCA scores
    plt.figure(figsize=(8, 6))

    plt.scatter(arg_x, arg_y, alpha=0.7, label="ARGPCA", color="blue")
    plt.scatter(pca_x, pca_y, alpha=0.7, label="PCA", color="red")

    plt.xlabel("PC1 score")
    plt.ylabel("PC2 score")
    plt.title("ARGPCA vs PCA scores on NASDAQ 2024-12 log returns")
    plt.legend()
    plt.tight_layout()

    # 6) Save figure
    if fig_path is None:
        fig_path = DEFAULT_FIG_PATH
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_path, dpi=150)

    print(f"Saved plot to: {fig_path}")

    # Optionally show the figure in interactive sessions
    plt.show()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the ARGPCA vs PCA real-data example with default settings."""
    run_argpca_and_plot(
        n_components=2,
        log_returns_path=DEFAULT_LOGRETURNS_CSV,
        mean_returns_path=DEFAULT_MEANRETURNS_CSV,
        fig_path=DEFAULT_FIG_PATH,
        align_signs=True,
    )


if __name__ == "__main__":
    main()