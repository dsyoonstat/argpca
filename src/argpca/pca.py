# pca.py
#
# ARG PCA: function-level API + scikit-learn style estimator.
#
# Public API:
#   - compute_arg_pc_subspace
#   - ARGPCA
#
# All functions assume samples have shape (n_samples, n_features) = (n, p),
# and we focus on HDLSS settings with p >> n.

from __future__ import annotations

from typing import Optional

import numpy as np
import warnings

from .utils import GramSpectrum, compute_gram_spectrum, recover_spike_directions


# ---------------------------------------------------------------------------
# Helper: orthonormalize reference vectors (rows) with rank detection
# ---------------------------------------------------------------------------


def _orthonormalize_reference_rows(
    reference_vectors: np.ndarray,
    p: int,
) -> np.ndarray:
    """
    Orthonormalize reference vectors given as rows in R^p, with rank detection.

    Parameters
    ----------
    reference_vectors : (r, p) array_like
        Each row is a reference vector v_j in R^p.
    p : int
        Feature dimension (for sanity checks).

    Returns
    -------
    V_orth_rows : (r_eff, p) ndarray
        Orthonormal basis of the span of the reference vectors in feature
        space, stored as **rows**. That is, each row is a unit vector in R^p
        and rows are mutually orthogonal:
            V_orth_rows @ V_orth_rows.T = I_{r_eff}.

        The span of these rows equals the span of the input rows.

    Notes
    -----
    - If the input rows are linearly dependent, numerically dependent
      directions are dropped and a warning is issued.
    """
    V_ref = np.asarray(reference_vectors)
    if V_ref.ndim != 2:
        raise ValueError("`reference_vectors` must be 2D with shape (r, p).")

    r, p_ref = V_ref.shape
    if p_ref != p:
        raise ValueError(
            f"`reference_vectors` must have shape (r, p={p}), "
            f"but got (r, {p_ref})."
        )
    if r == 0:
        raise ValueError("`reference_vectors` must contain at least one row.")

    # Work in feature space: V_feat = V_ref^T ∈ R^{p×r}
    V_feat = V_ref.T  # (p, r)

    # QR with reduced mode: Q_feat (p, r), R (r, r)
    # Columns of Q_feat are orthonormal in R^p.
    Q_feat, R = np.linalg.qr(V_feat, mode="reduced")

    diag_R = np.abs(np.diag(R))
    if diag_R.size == 0:
        raise ValueError(
            "All reference vectors are numerically zero; "
            "cannot construct a reference subspace."
        )

    # Numerical rank tolerance (standard QR-based heuristic)
    tol = diag_R.max() * max(p, r) * np.finfo(R.dtype).eps
    independent_mask = diag_R > tol
    r_eff = int(independent_mask.sum())

    if r_eff == 0:
        raise ValueError(
            "Reference vectors are numerically rank-zero after QR; "
            "cannot construct a reference subspace."
        )

    if r_eff < r:
        dropped = r - r_eff
        warnings.warn(
            f"`reference_vectors` are linearly dependent: "
            f"rank={r_eff} < r={r}. "
            f"Dropped {dropped} dependent direction(s) and "
            "replaced them with a linearly independent orthonormal basis "
            "spanning the same row span.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Columns of Q_feat form an orthonormal basis in R^p.
    # We select the independent ones and transpose to get **row-orthonormal**
    # basis: shape (r_eff, p).
    V_orth_rows = Q_feat[:, independent_mask].T  # (r_eff, p)
    return V_orth_rows


# ---------------------------------------------------------------------------
# Function-level API
# ---------------------------------------------------------------------------


def compute_arg_pc_subspace(
    samples: np.ndarray,
    reference_vectors: np.ndarray,
    n_components: int,
    orthonormal: bool = True,
    spectrum: Optional[GramSpectrum] = None,
) -> np.ndarray:
    """
    Compute ARG (Adaptive Reference Guided) PC subspace basis in feature space.

    This is the core functional API for ARG PCA. It uses the Gram-matrix
    trick to operate efficiently in HDLSS (p >> n) regimes and can reuse
    a precomputed GramSpectrum to avoid repeated eigendecompositions.

    Parameters
    ----------
    samples : (n, p) array_like
        Data matrix with rows = samples, columns = features.
    reference_vectors : (r, p) array_like
        Reference vectors in R^p. Each **row** is a reference vector v_j.
        They may be linearly dependent; internally we detect linear
        dependencies, reduce to a linearly independent orthonormal basis
        (row-orthonormal), and issue a warning if some directions are dropped.
    n_components : int
        Number of leading components / spikes (subspace dimension m).
    orthonormal : bool, default True
        If True, orthonormalize the final ARG subspace basis.
    spectrum : GramSpectrum, optional
        Precomputed Gram spectrum for `samples`. If provided, must have been
        computed from the same data matrix. If None, this function computes
        the spectrum internally.

    Returns
    -------
    components_arg : (m, p) ndarray
        Estimated ARG PC basis in feature space. Each row is a component
        (direction) in R^p. This matches scikit-learn's
        ``PCA.components_`` convention.

    Notes
    -----
    - This function returns a basis for the ARG-guided subspace.
      It does **not** perform a second PCA inside that subspace; that
      step is handled by the ARGPCA estimator.
    - When `spectrum` is provided, eigendecomposition of the Gram matrix is
      not recomputed. This is useful inside ARGPCA.fit, where the same
      spectrum is reused to derive multiple quantities.
    """
    X = np.asarray(samples)
    if X.ndim != 2:
        raise ValueError(f"`samples` must be 2D, got {X.ndim}D.")
    n, p = X.shape
    if n < 2:
        raise ValueError("Need at least two samples (n >= 2).")

    # 1) Gram spectrum (center + eigendecomposition), possibly reused
    if spectrum is None:
        spectrum = compute_gram_spectrum(X)

    spectrum.check_n_components(n_components)

    # 2) Recover spike eigenvectors and bulk level
    #    U_spike : (p, m), lam_spike : (m,), l_tilde : scalar
    U_spike, lam_spike, l_tilde = recover_spike_directions(
        spectrum, n_components
    )
    m = n_components

    # 3) Orthonormalize reference rows and apply (I - P_V) to U_spike
    #
    #    V_orth_rows : (r_eff, p) with row-orthonormal basis.
    #    Projection onto reference subspace:
    #        P_V = V^T V
    #    So:
    #        (I - P_V) U_spike = U_spike - V^T (V U_spike)
    #
    V_orth_rows = _orthonormalize_reference_rows(reference_vectors, p=p)  # (r_eff, p)
    VU = V_orth_rows @ U_spike                                           # (r_eff, m)
    M = U_spike - V_orth_rows.T @ VU                                     # (p, m)

    # 4) Compute U_ARG = (S_m - l_tilde I_p) M
    #    where S_m = sum_{i=1}^m lam_i u_i u_i^T.
    #
    #    Implementation via:
    #      UtM   = U_spike^T M   ∈ R^{m×m}
    #      S_m_M = U_spike (diag(lam_spike) UtM) ∈ R^{p×m}
    UtM = U_spike.T @ M                              # (m, m)
    S_m_M = U_spike @ (lam_spike[:, None] * UtM)     # (p, m)
    U_arg = S_m_M - l_tilde * M                      # (p, m)

    # 5) Optional orthonormalization (columns in feature space)
    if orthonormal:
        U_arg, _ = np.linalg.qr(U_arg, mode="reduced")  # (p, m)

    # Return in (m, p) convention (rows = components), sklearn-style
    return U_arg.T


# ---------------------------------------------------------------------------
# Estimator-style API (scikit-learn compatible)
# ---------------------------------------------------------------------------


class ARGPCA:
    """
    ARG PCA estimator with a scikit-learn–style interface.

    Conceptually, this estimator performs:

    1. Estimate an ARG-guided subspace in feature space using
       :func:`compute_arg_pc_subspace`.
    2. Project the centered data onto this subspace.
    3. Perform an ordinary PCA *within* the projected subspace.

    This matches the description “project onto ARG PC subspace, then do PCA
    again on the projected data”.

    Parameters
    ----------
    n_components : int
        Number of leading ARG components to retain (subspace dimension m).

    Notes
    -----
    - We assume HDLSS settings with p >> n.
    - The estimator requires reference vectors to be passed to ``fit``.
    """

    def __init__(self, n_components: int):
        self.n_components = int(n_components)

        # Attributes set during fit
        self.n_samples_: Optional[int] = None
        self.n_features_: Optional[int] = None
        self.n_components_: Optional[int] = None

        self.mean_: Optional[np.ndarray] = None                 # (p,)
        self.components_: Optional[np.ndarray] = None           # (m, p)
        self.explained_variance_: Optional[np.ndarray] = None   # (m,)
        self.explained_variance_ratio_: Optional[np.ndarray] = None  # (m,)
        self.noise_variance_: Optional[float] = None

        # Internal (not part of public API, but useful conceptually)
        self._arg_basis_: Optional[np.ndarray] = None           # (m, p)

    # ---------------------------- core methods ---------------------------- #

    def fit(
        self,
        X: np.ndarray,
        reference_vectors: np.ndarray,
        y=None,
    ) -> "ARGPCA":
        """
        Fit the ARG PCA model on X given reference vectors.

        Parameters
        ----------
        X : (n, p) array_like
            Data matrix (rows = samples, columns = features).
        reference_vectors : (r, p) array_like
            Reference vectors in R^p. Each row is a reference vector.
        y : Ignored
            Exists for scikit-learn compatibility.

        Returns
        -------
        self : ARGPCA
            Fitted estimator.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"`X` must be 2D, got {X.ndim}D.")
        n, p = X.shape
        if n < 2:
            raise ValueError("Need at least two samples (n >= 2).")

        m = self.n_components
        if not isinstance(m, int) or m <= 0:
            raise ValueError("`n_components` must be a positive integer.")

        # 1) Store basic stats
        self.n_samples_ = n
        self.n_features_ = p
        self.n_components_ = m

        # 2) Compute Gram spectrum once (centering inside)
        spectrum = compute_gram_spectrum(X)
        spectrum.check_n_components(m)

        # 3) Compute ARG subspace basis using the shared spectrum
        #    components_arg : (m, p) with orthonormal rows.
        components_arg = compute_arg_pc_subspace(
            samples=X,
            reference_vectors=reference_vectors,
            n_components=m,
            orthonormal=True,
            spectrum=spectrum,
        )  # (m, p)

        # Store ARG basis (row-wise)
        self._arg_basis_ = components_arg

        # 4) Project centered data onto ARG subspace
        #    Use the same centering as in compute_gram_spectrum:
        #    spectrum.Xc is X - X.mean(axis=0).
        Xc = spectrum.Xc                     # (n, p)
        B = components_arg.T                 # (p, m), columns = ARG basis
        Z = Xc @ B                           # (n, m)

        # 5) Perform PCA within ARG subspace:
        #    Covariance in ARG coordinates: S_H = (1/n) Z^T Z  (m × m)
        S_H = (Z.T @ Z) / float(n)           # (m, m), symmetric PSD
        eigvals, eigvecs = np.linalg.eigh(S_H)  # ascending
        idx = np.argsort(eigvals)[::-1]         # descending
        eigvals = eigvals[idx]                  # (m,)
        eigvecs = eigvecs[:, idx]               # (m, m), cols = eigvecs

        # 6) Map eigenvectors back to feature space:
        #    If a_j is the j-th eigenvector in R^m, then
        #       v_j = B a_j ∈ R^p
        #    We stack them into V = B A with A = eigvecs.
        V = B @ eigvecs                    # (p, m), columns are components
        # Orthonormality should hold numerically; we keep the shapes consistent
        self.components_ = V.T             # (m, p), rows = components

        # 7) Mean and variance-related attributes
        self.mean_ = X.mean(axis=0)        # same centering convention

        # Explained variance = eigenvalues of covariance in ARG subspace
        self.explained_variance_ = eigvals  # (m,)

        # Explained variance ratio relative to total variance of X
        k = spectrum.rank
        total_var = float(np.sum(spectrum.eigvals[:k]))
        if total_var > 0.0:
            self.explained_variance_ratio_ = eigvals / total_var
        else:
            self.explained_variance_ratio_ = np.zeros_like(eigvals)

        # Noise variance: mean of non-spike eigenvalues of full S
        #   noise_variance = mean(lambda_{m+1}, ..., lambda_{n-1}) * n / p,
        # where lambda_i are eigenvalues of sample covariance.
        if k > m:
            bulk = spectrum.eigvals[m:k]
            mean_bulk = float(bulk.mean())
            self.noise_variance_ = mean_bulk * float(n) / float(p)
        else:
            self.noise_variance_ = 0.0

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project X onto the learned ARG principal components.

        Parameters
        ----------
        X : (n, p) array_like
            New data matrix.

        Returns
        -------
        X_transformed : (n, m) ndarray
            ARG PC scores (coordinates in the final ARG PCA directions).
        """
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("ARGPCA instance is not fitted yet.")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"`X` must be 2D, got {X.ndim}D.")
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected X with {self.n_features_} features, "
                f"but got {X.shape[1]}."
            )

        Xc = X - self.mean_
        # components_: (m, p) → components_.T: (p, m)
        return Xc @ self.components_.T

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from ARG PC scores.

        Parameters
        ----------
        X_transformed : (n, m) array_like
            ARG PC scores.

        Returns
        -------
        X_reconstructed : (n, p) ndarray
            Approximate reconstruction in the original feature space.
        """
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("ARGPCA instance is not fitted yet.")

        Z = np.asarray(X_transformed)
        if Z.ndim != 2:
            raise ValueError(f"`X_transformed` must be 2D, got {Z.ndim}D.")
        if Z.shape[1] != self.n_components_:
            raise ValueError(
                f"Expected `X_transformed` with {self.n_components_} "
                f"components, but got {Z.shape[1]}."
            )

        # Z (n, m) @ components_ (m, p) + mean_ (p,)
        return Z @ self.components_ + self.mean_

    def fit_transform(
        self,
        X: np.ndarray,
        reference_vectors: np.ndarray,
        y=None,
    ) -> np.ndarray:
        """
        Fit the model to X and return the ARG PC scores.

        Equivalent to calling ``fit(X, reference_vectors).transform(X)``.

        Parameters
        ----------
        X : (n, p) array_like
        reference_vectors : (r, p) array_like
        y : Ignored

        Returns
        -------
        X_transformed : (n, m) ndarray
        """
        return self.fit(X, reference_vectors, y=y).transform(X)

    # -------------------------- optional helper --------------------------- #

    def get_covariance(self) -> np.ndarray:
        """
        Estimate the covariance matrix implied by the fitted ARG PCA model.

        Returns
        -------
        cov : (p, p) ndarray
            Estimated covariance:
                cov = components_.T @ diag(explained_variance_ - noise_variance_) @ components_
                      + noise_variance_ * I_p

        Notes
        -----
        - This constructs a full (p, p) matrix, which may be large when
          p is very high. Use with care in HDLSS settings.
        """
        if (
            self.components_ is None
            or self.explained_variance_ is None
            or self.noise_variance_ is None
        ):
            raise RuntimeError("ARGPCA instance is not fitted yet.")

        p = self.n_features_
        components = self.components_              # (m, p)
        ev = self.explained_variance_             # (m,)
        noise = float(self.noise_variance_)

        # Signal part: components^T diag(ev - noise) components
        diag_signal = ev - noise
        signal = components.T @ (diag_signal[:, None] * components)  # (p, p)

        cov = signal + noise * np.eye(p, dtype=components.dtype)
        return cov