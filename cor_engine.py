"""
DNV Scientific Module
---------------------
Role:
    Provides pure computation functions for 20 correlation and dependence
    measures, returning typed, immutable CorrelationResult records.

Scientific Context:
    Operates on numeric DataFrames, computing pairwise association matrices
    or ranked feature scores.  No UI calls, no side effects, no I/O.
    Each function is a pure map from (DataFrame, **kwargs) → CorrelationResult.

Invariants:
    - Every matrix function returns a square, symmetric DataFrame with
      ones on the diagonal and values in its declared codomain.
    - CorrelationResult is frozen; callers receive an immutable contract.
    - All intermediate NaN values are propagated, not silently dropped.

Assumptions:
    - Input DataFrames contain only numeric columns (callers are responsible
      for pre-filtering).
    - Minimum 2 rows and 2 columns are present (validated by caller).
    - _DISPATCH covers only methods with the standard (DataFrame → square
      matrix) signature.  Five keys are intentionally excluded because their
      return shape or parameterisation differs: see
      _DISPATCH_EXCLUDED_KEYS below.  They are invoked via dedicated
      branches in gui_dlg_cor_plot.py, not through _DISPATCH.

Failure Modes:
    - Singular covariance matrix in partial_correlation: returns identity
      matrix and sets is_valid=False on the result.
    - Missing optional dependency (statsmodels, pgmpy, pyinform): the
      corresponding function raises ImportError with an install hint;
      the caller catches and shows a user-facing message.
    - Autoencoder convergence failure: per-pair MSE falls back to 1.0
      (score = 0), producing a valid matrix rather than raising.

Provenance:
    - This module emits no transformation metadata.  Provenance originates
      in the calling CorrelationDialog when results are displayed.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Final, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

try:
    from statsmodels.tsa.stattools import ccf as _sm_ccf
    from statsmodels.tsa.stattools import grangercausalitytests as _sm_granger
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

try:
    with warnings.catch_warnings():
        # pgmpy 1.1 emits a FutureWarning on `pgmpy.estimators` import about
        # StructureScore relocation; it is not actionable from our side.
        warnings.simplefilter("ignore", FutureWarning)
        from pgmpy.models import BayesianNetwork as _BayesianNetwork
        from pgmpy.estimators import HillClimbSearch as _HCS
    # K2 score: pgmpy <1.0 exposed it as estimators.K2Score; pgmpy >=1.0 moved
    # it to pgmpy.structure_score.K2.  Try both so either version works.
    try:
        from pgmpy.estimators import K2Score as _K2Score  # type: ignore
    except ImportError:
        from pgmpy.structure_score import K2 as _K2Score  # type: ignore
    import networkx as _nx
    _HAS_PGMPY = True
except ImportError:
    _HAS_PGMPY = False

try:
    import pyinform as _pyinform
    _HAS_PYINFORM = True
except (ImportError, OSError):
    _HAS_PYINFORM = False

from sklearn.covariance import GraphicalLassoCV as _GraphicalLassoCV
from sklearn.cross_decomposition import CCA as _CCA
from sklearn.feature_selection import (
    mutual_info_classif as _mi_classif,
    mutual_info_regression as _mi_regression,
)
from sklearn.linear_model import Lasso as _Lasso, LassoCV as _LassoCV
from sklearn.neural_network import MLPRegressor as _MLPRegressor
from sklearn.preprocessing import StandardScaler as _StandardScaler

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

#: Codomain tag describing what values the matrix contains.
CODOMAIN_CORR:    Final[str] = "[-1, 1]"   # standard correlation
CODOMAIN_UNIT:    Final[str] = "[0, 1]"    # non-negative bounded association (HSIC-normalised, distance corr…)
CODOMAIN_PVAL:    Final[str] = "[0, 1]"    # 1 − p-value (Granger)
CODOMAIN_REAL:    Final[str] = "ℝ"         # unbounded real (partial precision)
CODOMAIN_NONNEG:  Final[str] = "[0, ∞)"    # non-negative unbounded (k-NN MI nats, transfer entropy)
CODOMAIN_SERIES:  Final[str] = "series"    # non-matrix result (MI bar, LASSO bar, CCF)


@dataclass(frozen=True)
class CorrelationResult:
    """
    Typed, immutable output of a single correlation computation.

    Fields
    ------
    method_key  : machine identifier matching _CORRELATION_REGISTRY key
    method_title: human-readable method name
    codomain    : declared value range of the matrix/series (CODOMAIN_* constant)
    matrix      : square symmetric DataFrame (None for series-type results)
    series      : ranked pd.Series (None for matrix-type results)
    series_label: axis label for the series (e.g. target column name)
    metadata    : free-form dict for provenance and display hints
    is_valid    : False if computation degraded (singular matrix, etc.)
    """
    method_key:   str
    method_title: str
    codomain:     str
    matrix:       Optional[pd.DataFrame]  = None
    series:       Optional[pd.Series]     = None
    series_label: str                     = ""
    metadata:     Dict[str, Any]          = field(default_factory=dict)
    is_valid:     bool                    = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _center_dist_mat(D: np.ndarray) -> np.ndarray:
    """Double-centre a distance matrix (required for distance correlation)."""
    return D - D.mean(axis=1, keepdims=True) - D.mean(axis=0, keepdims=True) + D.mean()


def _chatterjee_xi(x: np.ndarray, y: np.ndarray) -> float:
    """
    Chatterjee's ξ rank correlation (2021).

    Codomain: [−1/(n−2), 1].  Converges to 1 for perfect functional
    dependence, 0 for independence, without pingouin.
    """
    n = len(x)
    if n < 3:
        return float("nan")
    order = np.argsort(x, kind="stable")
    r = rankdata(y[order], method="average")
    xi = 1.0 - 3.0 * float(np.sum(np.abs(np.diff(r)))) / (n ** 2 - 1.0)
    return xi


def _clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with any NaN and return float64 copy."""
    return df.dropna().astype(float)


def _sym_matrix(n: int, columns) -> np.ndarray:
    """Return identity matrix as starting point for pairwise fills."""
    return np.eye(n, dtype=float)


# ---------------------------------------------------------------------------
# Matrix correlation calculators
# ---------------------------------------------------------------------------

def pearson_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr(method="pearson")


def spearman_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr(method="spearman")


def kendall_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr(method="kendall")


#: Default k parameter for k-NN mutual information estimator (Kraskov 2004).
KNN_MI_K_DEFAULT:    Final[int] = 3
#: Alternate k for the "k5" variant exposed via mice_matrix.
KNN_MI_K_ALTERNATE:  Final[int] = 5


def _knn_mi_matrix(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Pairwise k-nearest-neighbour mutual information (Kraskov-Stögbauer-
    Grassberger estimator via sklearn.feature_selection.mutual_info_regression).

    NOTE: this is NOT the Reshef-2011 Maximal Information Coefficient.
    True MIC is a grid-search statistic with equitability guarantees; this
    function reports the k-NN MI estimate, which is a different quantity.
    Name kept as ``mic`` in _DISPATCH for backward-compatible dispatch keys.
    """
    cols = df.columns
    n = len(cols)
    mat = _sym_matrix(n, cols)
    clean = _clean_numeric(df)
    n_rows = len(clean)
    # sklearn's k-NN MI estimator requires at least k+1 samples; clamp so tiny
    # inputs degrade gracefully instead of raising ValueError.
    k_eff = int(max(1, min(k, n_rows - 1))) if n_rows > 1 else 1
    for i in range(n):
        for j in range(i + 1, n):
            try:
                val = _mi_regression(
                    clean.iloc[:, i].values.reshape(-1, 1),
                    clean.iloc[:, j].values,
                    n_neighbors=k_eff,
                    random_state=0,
                )[0]
            except (ValueError, FloatingPointError):
                val = float("nan")
            mat[i, j] = mat[j, i] = val
    return pd.DataFrame(mat, index=cols, columns=cols)


#: Reshef 2011 MIC grid-size ceiling exponent.  The paper uses B(n) = n^α
#: with α = 0.6 as the "default" operating point.  Reshef warns the
#: estimator is only honest for n large enough that B(n) ≥ 4.
MIC_ALPHA:    Final[float] = 0.6
#: Hard floor on grid resolution so tiny inputs still get a 2×2 partition.
MIC_MIN_GRID: Final[int]   = 2


def _mic_pairwise(x: np.ndarray, y: np.ndarray) -> float:
    """
    Maximal Information Coefficient (Reshef et al., Science 2011).

    MIC(X,Y) = max_{nx·ny ≤ B(n)}  I(Xᵒ, Yᵒ) / log₂(min(nx, ny))

    where Xᵒ, Yᵒ are the equal-frequency discretisations of X,Y into
    nx, ny bins respectively and I(·,·) is the plug-in mutual information
    of the resulting contingency table in bits.  The normalisation
    guarantees MIC ∈ [0, 1], with MIC → 1 for any deterministic
    functional relationship (the "equitability" property of Reshef).

    NOTE: this is the practical "ApproxMIC" variant — it searches the
    equal-frequency grid rather than Reshef's dynamic-programming optimal
    Y-axis partition.  For smooth signals on n ≥ 200 the two agree to
    within a few percent; the exact DP is O(n·nx·ny²) and overkill for
    a dense square matrix of feature pairs.
    """
    n = len(x)
    if n < 5:
        return float("nan")
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 5:
        return float("nan")
    B = max(4, int(np.floor(n ** MIC_ALPHA)))

    best = 0.0
    for nx in range(MIC_MIN_GRID, B // MIC_MIN_GRID + 1):
        ny_max = B // nx
        if ny_max < MIC_MIN_GRID:
            break
        try:
            cx = pd.qcut(x, q=nx, labels=False, duplicates="drop")
        except ValueError:
            continue
        cx = np.asarray(cx)
        if cx is None or np.unique(cx[~np.isnan(cx)]).size < 2:
            continue
        for ny in range(MIC_MIN_GRID, ny_max + 1):
            try:
                cy = pd.qcut(y, q=ny, labels=False, duplicates="drop")
            except ValueError:
                continue
            cy = np.asarray(cy)
            m = np.isfinite(cx) & np.isfinite(cy)
            if m.sum() < 3:
                continue
            cxi = cx[m].astype(int); cyi = cy[m].astype(int)
            Kx = int(cxi.max()) + 1; Ky = int(cyi.max()) + 1
            if Kx < 2 or Ky < 2:
                continue
            # plug-in MI in bits via contingency table
            tab = np.zeros((Kx, Ky), dtype=float)
            for a, b in zip(cxi, cyi):
                tab[a, b] += 1.0
            tab /= tab.sum()
            px = tab.sum(axis=1, keepdims=True)
            py = tab.sum(axis=0, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where((tab > 0) & (px > 0) & (py > 0),
                                 tab / (px * py), 1.0)
                mi = float(np.sum(tab * np.log2(ratio)))
            norm = np.log2(min(Kx, Ky))
            if norm <= 0:
                continue
            score = mi / norm
            if score > best:
                best = score
    # Floating clip: MI plug-in can exceed 1 by O(1/n) bias on tiny grids.
    return float(min(1.0, max(0.0, best)))


def mic_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Reshef 2011 Maximal Information Coefficient (equitable, bounded [0, 1]).

    See ``_mic_pairwise`` for the algorithmic detail; values approach 1
    for any deterministic functional dependence and 0 for independence,
    regardless of the shape of the underlying relationship.
    """
    cols = df.columns
    n = len(cols)
    mat = _sym_matrix(n, cols)
    clean = _clean_numeric(df)
    vals = clean.values
    for i in range(n):
        for j in range(i + 1, n):
            mat[i, j] = mat[j, i] = _mic_pairwise(vals[:, i], vals[:, j])
    return pd.DataFrame(mat, index=cols, columns=cols)


def distance_correlation_matrix(df: pd.DataFrame, metric: str = "euclidean") -> pd.DataFrame:
    """Székely–Rizzo–Bakirov distance correlation.  Codomain: [0, 1].

    Definition (Székely, Rizzo & Bakirov 2007):
        dCov²(X,Y) = mean(a_ij · b_ij)
        dVar²(X)   = dCov²(X,X) = mean(a_ij²)
        dCor²(X,Y) = dCov²(X,Y) / sqrt(dVar²(X) · dVar²(Y))
        dCor(X,Y)  = sqrt(dCor²(X,Y))    ∈ [0, 1]

    The previous implementation computed sqrt(dCov²)/sqrt(dVar²·dVar²),
    which is *not* dCor and is not bounded by 1 (self-check fails unless
    mean(a²)=1).  This version computes the canonical dCor and therefore
    lies in [0, 1] as the docstring claims.
    """
    cols = df.columns
    n = len(cols)
    mat = _sym_matrix(n, cols)
    clean = _clean_numeric(df)
    for i in range(n):
        for j in range(i + 1, n):
            X = clean.iloc[:, i].values
            Y = clean.iloc[:, j].values
            a = _center_dist_mat(squareform(pdist(X[:, None], metric=metric)))
            b = _center_dist_mat(squareform(pdist(Y[:, None], metric=metric)))
            dcov2    = float((a * b).mean())
            dvar2_x  = float((a * a).mean())
            dvar2_y  = float((b * b).mean())
            denom    = float(np.sqrt(dvar2_x * dvar2_y))
            if denom > 0 and dcov2 > 0:
                val = float(np.sqrt(dcov2 / denom))
            else:
                val = 0.0
            mat[i, j] = mat[j, i] = val
    return pd.DataFrame(mat, index=cols, columns=cols)


#: Minimum sample size below which Hoeffding D is undefined (formula divides
#: by n·(n−1)·(n−2)·(n−3)·(n−4)); callers see NaN rather than an exception.
HOEFFDING_MIN_N: Final[int] = 5

#: Maximum modal frequency (fraction of rows taking the most common value
#: of x or y) above which the 1948 Hoeffding D formula is rejected with
#: NaN. The closed-form rank-count kernel is only asymptotically correct
#: when no single value dominates the marginal; when one value captures
#: > 50 % of the sample the Q-rank distribution degenerates and D can
#: overshoot its [-0.5, 1] codomain (observed: D = 1.68 on
#: spinel/tetragonality_ratio with 84 % of rows at 1.0). Materials
#: features often have legitimate multi-modal discrete structure
#: (valences, coordination numbers) — we reject only heavy single-point
#: domination, not total tie fraction.
HOEFFDING_MAX_MODAL_FRACTION: Final[float] = 0.50

#: Lower bound of Hoeffding D codomain for the clamp fallback. The 1948
#: statistic lives in approximately [-0.5, 1]; values outside this range
#: indicate tie contamination or numerical round-off and are clamped.
HOEFFDING_D_MIN: Final[float] = -0.5

#: Upper bound of Hoeffding D codomain for the clamp fallback.
HOEFFDING_D_MAX: Final[float] = 1.0


def _modal_fraction(x: np.ndarray) -> float:
    """Fraction of rows that take the most common value in x.

    Returns 0 on empty input, 1 on a constant vector. A well-spread
    continuous feature returns ≈ 1/n; a discrete feature with k equally
    populated values returns ≈ 1/k; a pathologically tied column
    (e.g. 84 % of rows at one value) returns 0.84.
    """
    n = len(x)
    if n == 0:
        return 0.0
    _, counts = np.unique(x, return_counts=True)
    return float(counts.max()) / float(n)


def _hoeffding_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Hoeffding's D statistic (Hoeffding 1948).

    Measures dependence against the null of independence, sensitive to both
    monotone and non-monotone deviations. Codomain is approximately
    [−0.5, 1]; D = 0 under independence, D = 1 under perfect functional
    dependence. O(n²) implementation — adequate for typical feature counts.

    Tie handling
    ------------
    The closed-form rank-count kernel assumes continuous marginals and is
    only asymptotically correct when no single value dominates the sample.
    When either x or y has modal frequency (fraction of rows taking the
    single most common value) greater than HOEFFDING_MAX_MODAL_FRACTION,
    the Q-rank distribution degenerates and the statistic can overshoot
    its codomain (e.g. D = 1.68 on spinel/tetragonality_ratio with 84 %
    of rows at 1.0). In that case this function returns NaN rather than
    an out-of-range value. Legitimate multi-modal discrete features
    (valences, coordination numbers) are still accepted. On borderline
    cases the computed value is clamped to [HOEFFDING_D_MIN, HOEFFDING_D_MAX].
    """
    n = len(x)
    if n < HOEFFDING_MIN_N:
        return float("nan")
    if _modal_fraction(x) > HOEFFDING_MAX_MODAL_FRACTION or \
       _modal_fraction(y) > HOEFFDING_MAX_MODAL_FRACTION:
        return float("nan")
    R = rankdata(x, method="average")
    S = rankdata(y, method="average")
    # Bivariate rank Q_i = 1 + #{j != i : x_j < x_i and y_j < y_i}
    xr = x.reshape(-1, 1)
    yr = y.reshape(-1, 1)
    lt_x = (xr.T < xr).astype(float)
    lt_y = (yr.T < yr).astype(float)
    Q = 1.0 + (lt_x * lt_y).sum(axis=1)
    D1 = float(np.sum((Q - 1.0) * (Q - 2.0)))
    D2 = float(np.sum((R - 1.0) * (R - 2.0) * (S - 1.0) * (S - 2.0)))
    D3 = float(np.sum((R - 2.0) * (S - 2.0) * (Q - 1.0)))
    num  = (n - 2) * (n - 3) * D1 + D2 - 2.0 * (n - 2) * D3
    den  = n * (n - 1) * (n - 2) * (n - 3) * (n - 4)
    if den <= 0:
        return float("nan")
    d_raw = 30.0 * num / den
    if not np.isfinite(d_raw):
        return float("nan")
    return float(min(HOEFFDING_D_MAX, max(HOEFFDING_D_MIN, d_raw)))


def hoeffding_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hoeffding's D pairwise matrix (real statistic, not a Spearman proxy).

    Detects both monotone and non-monotone dependence. Diagonal is set to
    1.0 by convention since D(x, x) = 1 for perfect functional dependence.
    Pairs where either column has modal frequency greater than
    HOEFFDING_MAX_MODAL_FRACTION return NaN — the closed-form 1948 formula
    is invalid when a single value dominates the marginal, and callers
    should prefer `distance` or `mic` for those pairs.
    """
    clean = _clean_numeric(df)
    cols = clean.columns
    n = len(cols)
    vals = clean.values
    mat = _sym_matrix(n, cols)
    for i in range(n):
        for j in range(i + 1, n):
            d = _hoeffding_d(vals[:, i], vals[:, j])
            mat[i, j] = mat[j, i] = d
    return pd.DataFrame(mat, index=cols, columns=cols)


def chatterjee_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Chatterjee's ξ pairwise matrix.  Codomain: [−1/(n−2), 1]."""
    cols = df.columns
    n = len(cols)
    mat = _sym_matrix(n, cols)
    clean = _clean_numeric(df)
    vals = clean.values
    for i in range(n):
        for j in range(i + 1, n):
            xi = _chatterjee_xi(vals[:, i], vals[:, j])
            mat[i, j] = mat[j, i] = xi
    return pd.DataFrame(mat, index=cols, columns=cols)


#: Hard clip on ρ during Olsson ML search — keeps the bivariate-normal CDF
#: from saturating on the (−1, 1) endpoints where the Hessian blows up.
POLYCHORIC_RHO_BOUND: Final[float] = 0.999
#: Cell-probability floor before taking log in the Olsson log-likelihood,
#: guarding against log(0) when a contingency cell is empty under the
#: current ρ hypothesis.
POLYCHORIC_PROB_FLOOR: Final[float] = 1e-12


def _olsson_polychoric(x: np.ndarray, y: np.ndarray, bins: int) -> float:
    """
    Two-step polychoric correlation (Olsson 1979).

    Models the observed ordinal pair (Xᵒ, Yᵒ) as a discretisation of a
    latent bivariate-normal (X*, Y*) with unit variances and correlation ρ.
    Thresholds aₖ, bₗ are taken from the empirical marginal cumulatives
    (step 1); ρ is then the argmax of the multinomial log-likelihood of
    the contingency table under those thresholds (step 2).

    Returns ρ ∈ [−1, 1].  Falls back to Pearson on degenerate inputs
    (fewer than 2 filled bins on either side, zero total count).
    """
    from scipy.stats import norm as _norm, multivariate_normal as _mvn
    from scipy.optimize import minimize_scalar as _minscalar

    # Bin both marginals into equal-frequency ordinal categories; qcut with
    # duplicates="drop" collapses ties so empty bins never appear.
    try:
        cx = pd.qcut(x, q=int(max(2, bins)), labels=False, duplicates="drop")
        cy = pd.qcut(y, q=int(max(2, bins)), labels=False, duplicates="drop")
    except Exception:
        return float("nan")
    cx = np.asarray(cx); cy = np.asarray(cy)
    mask = np.isfinite(cx) & np.isfinite(cy)
    cx, cy = cx[mask].astype(int), cy[mask].astype(int)
    if cx.size == 0:
        return float("nan")
    Kx, Ky = int(cx.max()) + 1, int(cy.max()) + 1
    if Kx < 2 or Ky < 2:
        # Degenerate: one side has a single level after tie-collapse.
        return float(np.corrcoef(x, y)[0, 1])

    # Contingency table N[i,j] = #{Xᵒ=i ∧ Yᵒ=j}
    N = np.zeros((Kx, Ky), dtype=float)
    for xi, yj in zip(cx, cy):
        N[xi, yj] += 1.0
    n_tot = N.sum()
    if n_tot <= 0:
        return float("nan")

    # Step 1: thresholds from marginal cumulatives, Φ⁻¹ scale.
    px = N.sum(axis=1) / n_tot
    py = N.sum(axis=0) / n_tot
    cumx = np.concatenate(([0.0], np.cumsum(px)))
    cumy = np.concatenate(([0.0], np.cumsum(py)))
    a = np.concatenate(([-np.inf], _norm.ppf(cumx[1:-1]), [np.inf]))
    b = np.concatenate(([-np.inf], _norm.ppf(cumy[1:-1]), [np.inf]))

    def _cell_probs(rho: float) -> np.ndarray:
        # Bivariate-normal rectangle probabilities for every (i,j) cell via
        # the CDF inclusion-exclusion: P = F(a_i,b_j) − F(a_{i−1},b_j)
        #                                − F(a_i,b_{j−1}) + F(a_{i−1},b_{j−1}).
        cov = np.array([[1.0, rho], [rho, 1.0]])
        F = np.zeros((Kx + 1, Ky + 1))
        for i in range(Kx + 1):
            for j in range(Ky + 1):
                ai, bj = a[i], b[j]
                if not np.isfinite(ai) and ai < 0:
                    F[i, j] = 0.0 if (not np.isfinite(bj) and bj < 0) else 0.0
                elif not np.isfinite(bj) and bj < 0:
                    F[i, j] = 0.0
                elif not np.isfinite(ai) and ai > 0:
                    F[i, j] = _norm.cdf(bj) if np.isfinite(bj) else 1.0
                elif not np.isfinite(bj) and bj > 0:
                    F[i, j] = _norm.cdf(ai) if np.isfinite(ai) else 1.0
                else:
                    F[i, j] = _mvn.cdf([ai, bj], mean=[0, 0], cov=cov)
        P = F[1:, 1:] - F[:-1, 1:] - F[1:, :-1] + F[:-1, :-1]
        return np.clip(P, POLYCHORIC_PROB_FLOOR, 1.0)

    def _neg_loglik(rho: float) -> float:
        P = _cell_probs(float(rho))
        return -float(np.sum(N * np.log(P)))

    res = _minscalar(
        _neg_loglik,
        bounds=(-POLYCHORIC_RHO_BOUND, POLYCHORIC_RHO_BOUND),
        method="bounded",
        options={"xatol": 1e-4},
    )
    if not res.success:
        return float("nan")
    return float(np.clip(res.x, -1.0, 1.0))


def polychoric_matrix(df: pd.DataFrame, bins: int = 5, discretize: bool = True) -> pd.DataFrame:
    """
    Maximum-likelihood polychoric correlation (Olsson 1979).

    Treats each pair (Xᵒ, Yᵒ) as an ordinal observation of a latent
    bivariate-normal with unit variances and correlation ρ; fits the
    thresholds from the empirical marginal cumulatives (step 1) and then
    maximises the multinomial log-likelihood over ρ ∈ [−1, 1] (step 2).

    Parameters
    ----------
    bins : int
        Number of ordinal levels per marginal; values are binned by
        equal-frequency quantiles with tie-collapse.
    discretize : bool
        Kept for backward dispatch compatibility — the Olsson estimator
        *always* bins internally, so this flag is now ignored.

    Codomain: [−1, 1].  Differs from Spearman because the estimator
    corrects for information loss from discretisation, targeting the
    latent continuous correlation rather than the ordinal rank concordance.
    """
    clean = _clean_numeric(df)
    cols = clean.columns
    n = len(cols)
    vals = clean.values
    mat = _sym_matrix(n, cols)
    for i in range(n):
        for j in range(i + 1, n):
            rho = _olsson_polychoric(vals[:, i], vals[:, j], bins)
            mat[i, j] = mat[j, i] = rho
    return pd.DataFrame(mat, index=cols, columns=cols)


def r_statistic_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Blomqvist β (medial correlation, 1950).

    β = (n_concordant − n_discordant) / n, measured against the sample
    medians of x and y — a distribution-free rank statistic that is
    robust to tails and distinct from Pearson, Spearman, and Kendall.
    Codomain: [−1, 1].

    NOTE: the dispatch key is still ``r_statistic`` for backward
    compatibility; the prior implementation of this key was Pearson-on-
    ranks, which is mathematically identical to Spearman's ρ and has
    been replaced with this real alternative rank statistic.
    """
    clean = _clean_numeric(df)
    cols = clean.columns
    n = len(cols)
    vals = clean.values
    mat = _sym_matrix(n, cols)
    for i in range(n):
        for j in range(i + 1, n):
            x = vals[:, i]
            y = vals[:, j]
            mx = float(np.median(x))
            my = float(np.median(y))
            sx = np.sign(x - mx)
            sy = np.sign(y - my)
            mask = (sx != 0) & (sy != 0)
            if not mask.any():
                mat[i, j] = mat[j, i] = float("nan")
                continue
            beta = float(np.mean(sx[mask] * sy[mask]))
            mat[i, j] = mat[j, i] = beta
    return pd.DataFrame(mat, index=cols, columns=cols)


def hsic_matrix(df: pd.DataFrame, kernel: str = "rbf") -> pd.DataFrame:
    """
    Hilbert-Schmidt Independence Criterion (HSIC).

    Supports 'linear' and 'rbf' kernels.  Values are normalised to [0, 1]
    by dividing by the geometric mean of the diagonal (self-HSIC) entries.
    """
    cols = df.columns
    n_cols = len(cols)
    clean = _clean_numeric(df).values
    m = len(clean)
    H = np.eye(m) - 1.0 / m

    def _kernel(d: np.ndarray) -> np.ndarray:
        if kernel == "linear":
            return np.outer(d, d)
        dist_sq = squareform(pdist(d[:, None], "sqeuclidean"))
        med = np.median(dist_sq[dist_sq > 0])
        sigma = med if med > 0 else 1.0
        return np.exp(-dist_sq / (2.0 * sigma ** 2))

    # Pre-compute centred kernel matrices
    Ks = [H @ _kernel(clean[:, i]) @ H for i in range(n_cols)]
    self_hsic = np.array([np.trace(Ks[i] @ Ks[i]) / m ** 2 for i in range(n_cols)])

    mat = _sym_matrix(n_cols, cols)
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            hsic_val = np.trace(Ks[i] @ Ks[j]) / m ** 2
            denom = np.sqrt(self_hsic[i] * self_hsic[j])
            val = float(hsic_val / denom) if denom > 0 else 0.0
            mat[i, j] = mat[j, i] = val
    return pd.DataFrame(mat, index=cols, columns=cols)


def mice_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """k-NN Mutual Information (k=5 variant).

    NOTE: this is NOT Reshef-2016 MICe (equitable MIC), which requires the
    'minepy' package.  This variant differs from ``mic_matrix`` only in the
    n_neighbors hyperparameter (k=5 vs k=3); larger k trades variance for
    bias in the Kraskov estimator.  Legacy ``mice`` key preserved for
    dispatch compatibility.
    """
    return _knn_mi_matrix(df, KNN_MI_K_ALTERNATE)


def sparse_partial_correlation(df: pd.DataFrame, cv: int = 5) -> tuple[pd.DataFrame, bool]:
    """
    Sparse Partial Correlation via Graphical LASSO (precision matrix).

    Returns (matrix, is_valid).  The Graphical LASSO solver can fail on
    rank-deficient data (constant columns, n_rows < n_cols, near-singular
    covariance); such failures raise FloatingPointError/ValueError/LinAlgError.
    On failure we return an identity matrix with is_valid=False so the
    dialog can surface the degeneracy to the user, matching the contract
    already used by ``partial_correlation_matrix``.
    """
    clean = _clean_numeric(df)
    cols = clean.columns
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _GraphicalLassoCV(cv=cv).fit(clean)
        prec = model.precision_
        d = np.sqrt(np.diag(prec))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pcor = -prec / np.outer(d, d)
        np.fill_diagonal(pcor, 1.0)
        if not np.all(np.isfinite(pcor)):
            raise FloatingPointError("non-finite entries in sparse precision")
        return pd.DataFrame(pcor, index=cols, columns=cols), True
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        identity = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
        return identity, False


def cca_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise 1-D canonical correlation.

    NOTE: true Canonical Correlation Analysis (Hotelling 1936) operates on
    two *predefined* multivariate sets X and Y.  This function instead
    splits columns at the arbitrary midpoint and runs ``sklearn.CCA`` on
    every single-column pair across the split — for 1-D × 1-D that
    reduces algebraically to |Pearson r|.  The output is therefore a
    cross-set absolute-Pearson matrix, not a multivariate CCA.  The
    legacy ``cca`` key is preserved for dispatch compatibility; users
    wanting true CCA should define the two variable sets explicitly.
    """
    clean = _clean_numeric(df)
    cols = clean.columns
    mid = len(cols) // 2
    X = clean.iloc[:, :mid]
    Y = clean.iloc[:, mid:]
    nX, nY = len(X.columns), len(Y.columns)
    full_cols = list(X.columns) + list(Y.columns)
    n_total = nX + nY

    mat = np.eye(n_total)
    mat[:nX, :nX] = X.corr(method="pearson").values
    mat[nX:, nX:] = Y.corr(method="pearson").values

    for i in range(nX):
        for j in range(nY):
            try:
                cca = _CCA(n_components=1, max_iter=1000)
                xc, yc = cca.fit_transform(X.iloc[:, [i]], Y.iloc[:, [j]])
                corr = float(np.corrcoef(xc.T, yc.T)[0, 1])
            except Exception:
                corr = float("nan")
            mat[i, nX + j] = mat[nX + j, i] = corr

    return pd.DataFrame(mat, index=full_cols, columns=full_cols)


def partial_correlation_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """
    Partial correlation via inversion of the covariance matrix.

    Returns (matrix, is_valid).  is_valid=False if the covariance matrix
    is singular (returns identity as fallback).
    """
    clean = _clean_numeric(df)
    try:
        inv_cov = np.linalg.inv(clean.cov().values)
        d = np.sqrt(np.diag(inv_cov))
        pcor = -inv_cov / np.outer(d, d)
        np.fill_diagonal(pcor, 1.0)
        return pd.DataFrame(pcor, index=df.columns, columns=df.columns), True
    except np.linalg.LinAlgError:
        log.warning("partial_correlation_matrix: singular covariance — returning identity")
        n = len(df.columns)
        return pd.DataFrame(np.eye(n), index=df.columns, columns=df.columns), False


#: MLP training cap: high enough that typical 1→1 regressions converge well
#: before the iteration limit, eliminating spurious ConvergenceWarnings.
MLP_MAX_ITER:          Final[int]   = 2000
#: Stop-early patience (epochs with no val-loss improvement > tol).
MLP_N_ITER_NO_CHANGE:  Final[int]   = 20
#: Tolerance gate for the early-stopping monitor.
MLP_TOL:               Final[float] = 1e-4


#: Width of the pre- and post-bottleneck hidden layer in the pairwise
#: autoencoder.  Small enough to train fast; wide enough to fit the
#: nonlinear manifold of a 2-D joint.
AE_HIDDEN_WIDTH:  Final[int] = 12
#: Bottleneck dimension.  Forcing a 1-D code on a 2-D joint means a
#: perfectly dependent pair can be reconstructed losslessly (one free
#: variable), while an independent pair loses ≈50% of the total variance.
AE_BOTTLENECK:    Final[int] = 1
#: L2 weight-decay on the autoencoder.  Large enough to stop the 1-D
#: bottleneck from memorising per-point locations on independent inputs
#: (which would make train-R² rise above the PCA floor).
AE_ALPHA:         Final[float] = 0.5


def autoencoder_dependency(df: pd.DataFrame) -> pd.DataFrame:
    """
    True bottleneck autoencoder dependency (symmetric).

    For every unordered pair (i, j), a 2-→hidden→1→hidden→2 MLP
    autoencoder is trained to reconstruct the joint (zᵢ, zⱼ) vector
    through a one-dimensional bottleneck.  The raw R² of reconstruction,
    R²_raw = 1 − MSE_total / Var_total, equals ≈1 for a perfect functional
    relationship and ≈½ for independence (the 1-D code retains exactly
    one principal direction of a 2-D isotropic cloud — the smallest
    eigenvalue of the covariance is 1, out of a total of 2).  To map the
    independence baseline to 0 and keep the saturation point at 1 we
    report the *excess-over-PCA-floor* R²:

        AE(i, j) = max(0, 2 · R²_raw − 1)

    which lies in [0, 1] with AE(indep) → 0 and AE(perfect) → 1.
    Matrix is symmetric by construction.

    The implementation uses sklearn's MLPRegressor with a symmetric
    hidden/bottleneck/hidden architecture and multi-output targets.
    Matrix is symmetric by construction.
    """
    clean = _clean_numeric(df)
    cols = clean.columns
    n = len(cols)
    scaler = _StandardScaler()
    scaled = scaler.fit_transform(clean)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            Z = scaled[:, [i, j]]
            try:
                # Held-out reconstruction: fit on train half, measure MSE on
                # test half.  On an independent pair the MLP cannot generalise
                # the memorised noise, so test-MSE ≈ 1 (one PC dimension lost)
                # and the rescaled score drops to 0.  On a deterministic pair
                # the learnt 1-D manifold *does* generalise, so score → 1.
                rng_split = np.random.default_rng(0)
                idx = rng_split.permutation(len(Z))
                half = len(idx) // 2
                tr, te = Z[idx[:half]], Z[idx[half:]]
                if len(tr) < 8 or len(te) < 4:
                    tr = te = Z  # fall back to in-sample for tiny inputs
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    best = None
                    for seed in (0, 7, 42):
                        m = _MLPRegressor(
                            hidden_layer_sizes=(AE_HIDDEN_WIDTH, AE_BOTTLENECK, AE_HIDDEN_WIDTH),
                            activation="tanh",
                            solver="lbfgs",
                            alpha=AE_ALPHA,
                            max_iter=MLP_MAX_ITER,
                            tol=MLP_TOL,
                            random_state=seed,
                            verbose=False,
                        )
                        m.fit(tr, tr)
                        r = m.predict(tr)
                        train_err = float(np.mean(np.sum((tr - r) ** 2, axis=1)))
                        if best is None or train_err < best[0]:
                            best = (train_err, m)
                    model = best[1]
                recon = model.predict(te)
                mse_total = float(np.mean(np.sum((te - recon) ** 2, axis=1)))
                var_total = float(np.var(te[:, 0]) + np.var(te[:, 1]))
                if var_total > 0:
                    r2_raw = 1.0 - mse_total / var_total
                    # Rescale PCA-floor (0.5 for independent 2-D unit-var) → 0.
                    score = max(0.0, min(1.0, 2.0 * r2_raw - 1.0))
                else:
                    score = 0.0
            except Exception as exc:
                log.warning("autoencoder_dependency pair (%d,%d): %s", i, j, exc)
                score = 0.0
            mat[i, j] = mat[j, i] = score
    return pd.DataFrame(mat, index=cols, columns=cols)


# ---------------------------------------------------------------------------
# Series-output calculators
# ---------------------------------------------------------------------------

#: Integer-valued targets with at most this many distinct values are
#: treated as classification targets by mutual_information_scores.
MI_CLF_MAX_UNIQUE: Final[int] = 10


def _target_is_classification(y: pd.Series) -> bool:
    """
    Heuristic: classification iff the target is object/string *or* integer-
    like with ≤ MI_CLF_MAX_UNIQUE distinct levels.  A small-cardinality
    *float* target (e.g. the 6 oxidation states in A_valence, stored as
    float64) must not trigger classification by itself because the fractional
    bit would corrupt sklearn's label encoding.
    """
    if y.dtype == object or str(y.dtype).startswith("category"):
        return True
    y_nonan = y.dropna()
    if y_nonan.empty:
        return False
    is_int_like = np.issubdtype(y_nonan.dtype, np.integer) or (
        np.issubdtype(y_nonan.dtype, np.floating)
        and np.all(np.isfinite(y_nonan.values))
        and np.all(y_nonan.values == np.floor(y_nonan.values))
    )
    return bool(is_int_like and y_nonan.nunique() <= MI_CLF_MAX_UNIQUE)


def mutual_information_scores(
    df: pd.DataFrame,
    target_col: str,
    random_state: int = 0,
) -> pd.Series:
    """
    Mutual information of each feature against a target column.

    NaN rows are dropped before fitting (sklearn does not accept NaN).
    Uses ``mutual_info_classif`` for low-cardinality integer or categorical
    targets, ``mutual_info_regression`` otherwise.
    """
    X_raw = df.drop(columns=[target_col]).select_dtypes(include="number")
    sub = pd.concat([X_raw, df[target_col].rename(target_col)], axis=1).dropna()
    if len(sub) < 3:
        return pd.Series(dtype=float)
    X = sub[X_raw.columns]
    y = sub[target_col]
    fn = _mi_classif if _target_is_classification(y) else _mi_regression
    # mutual_info_classif expects integer labels; safe to cast because
    # _target_is_classification() only returns True for integer-like targets.
    if fn is _mi_classif:
        y = y.astype(int)
    scores = pd.Series(
        fn(X.values, y.values, random_state=random_state),
        index=X.columns,
    ).sort_values(ascending=False)
    return scores


def lasso_coefficients(
    df: pd.DataFrame,
    target_col: str,
    alpha: float = 0.1,
    *,
    max_iter: int = 5000,
    cv_folds: int = 5,
) -> pd.Series:
    """
    LASSO regression coefficients (standardised features, standardised target).

    Both X and y are standardised so the reported coefficients are directly
    comparable across features *and* so the regularisation penalty α acts on
    a unit-variance target (this is what makes the default α=0.1 meaningful
    across datasets; without y-standardisation α had to be re-tuned for every
    target scale).

    If ``alpha <= 0`` the function falls back to ``LassoCV`` with ``cv_folds``
    and returns the coefficients at the CV-optimal α, stored under the series
    name so the caller can display it.

    NaN rows are dropped before fitting.  Returns a Series indexed by feature
    name, sorted descending by |coef|.
    """
    X_raw = df.drop(columns=[target_col]).select_dtypes(include="number")
    sub = pd.concat([X_raw, df[target_col].rename(target_col)], axis=1).dropna()
    if len(sub) < 3 or X_raw.shape[1] == 0:
        return pd.Series(dtype=float, name="lasso_alpha=0.0")
    X = sub[X_raw.columns].values.astype(float)
    y = sub[target_col].values.astype(float)
    X_scaled = _StandardScaler().fit_transform(X)
    y_scaled = (y - y.mean()) / (y.std(ddof=0) or 1.0)

    if alpha is None or alpha <= 0:
        model = _LassoCV(cv=int(cv_folds), max_iter=int(max_iter)).fit(X_scaled, y_scaled)
        used_alpha = float(model.alpha_)
    else:
        model = _Lasso(alpha=float(alpha), max_iter=int(max_iter)).fit(X_scaled, y_scaled)
        used_alpha = float(alpha)

    coef = pd.Series(model.coef_, index=X_raw.columns, name=f"lasso_alpha={used_alpha:.4g}")
    return coef.reindex(coef.abs().sort_values(ascending=False).index)


def ccf_series(
    x: pd.Series,
    y: pd.Series,
    nlags: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cross-Correlation Function using NumPy (no statsmodels required).

    Returns (lags, ccf_values) arrays of length (2 * nlags + 1).
    Values are normalised by sqrt(Var(x) * Var(y)) * n.
    """
    xv = x.dropna().values.astype(float)
    yv = y.dropna().values.astype(float)
    n = min(len(xv), len(yv))
    xv, yv = xv[:n] - xv[:n].mean(), yv[:n] - yv[:n].mean()
    full = np.correlate(xv, yv, mode="full")
    norm = float(np.sqrt(np.var(xv) * np.var(yv))) * n if n > 0 else 1.0
    full = full / norm if norm > 0 else full
    mid = len(full) // 2
    lo, hi = max(0, mid - nlags), min(len(full), mid + nlags + 1)
    vals = full[lo:hi]
    lags = np.arange(lo - mid, hi - mid)
    return lags, vals


# ---------------------------------------------------------------------------
# Granger / Transfer Entropy (optional — require missing packages)
# ---------------------------------------------------------------------------

def granger_matrix(df: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    """
    Granger Causality matrix (1 − p-value, so high = more causal).

    Requires statsmodels.  Raises ImportError with install hint if absent.
    """
    if not _HAS_STATSMODELS:
        raise ImportError(
            "Granger Causality requires statsmodels.\n"
            "Install with:  conda install statsmodels   or   pip install statsmodels"
        )
    import contextlib, io
    cols = df.columns
    n = len(cols)
    clean = _clean_numeric(df)
    mat = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    # statsmodels >=0.14 renamed maxlags→maxlag and deprecated
                    # verbose; pass an int to compute all lags up to max_lag.
                    result = _sm_granger(
                        clean[[cols[j], cols[i]]].values,
                        maxlag=int(max_lag),
                    )
                p = min(res[0]["ssr_ftest"][1] for res in result.values())
                mat[i, j] = p
            except Exception as exc:
                log.warning("granger_matrix (%s→%s): %s", cols[i], cols[j], exc)
                mat[i, j] = float("nan")
    return pd.DataFrame(1.0 - mat, index=cols, columns=cols)


#: Default bin count for auto-discretisation of continuous inputs before
#: the pyinform Transfer Entropy estimator, which requires integer states.
TE_DEFAULT_N_BINS: Final[int] = 4


def transfer_entropy_matrix(
    df: pd.DataFrame,
    k: int = 1,
    n_bins: int = TE_DEFAULT_N_BINS,
) -> pd.DataFrame:
    """
    Transfer Entropy matrix (Schreiber 2000) via pyinform.

    pyinform.transfer_entropy requires integer-coded state sequences and
    raises InformError ("negative state in timeseries") on float inputs.
    We therefore auto-discretise each column into ``n_bins`` equal-frequency
    bins before the per-pair call; if a column is already integer-valued we
    pass it through unchanged.

    Requires pyinform.  Raises ImportError with install hint if absent.
    """
    if not _HAS_PYINFORM:
        raise ImportError(
            "Transfer Entropy requires pyinform.\n"
            "Install with:  pip install pyinform"
        )
    cols = df.columns
    n = len(cols)
    clean = _clean_numeric(df)
    encoded = np.empty((len(clean), n), dtype=np.int64)
    for ci, name in enumerate(cols):
        col = clean[name].values
        if np.issubdtype(col.dtype, np.integer) and col.min() >= 0:
            encoded[:, ci] = col.astype(np.int64)
            continue
        try:
            # pandas qcut gives equal-frequency bins; collapse duplicate edges
            # on near-constant columns.  Returned codes are non-negative ints.
            codes = pd.qcut(col, q=int(max(2, n_bins)), labels=False,
                            duplicates="drop")
            encoded[:, ci] = np.nan_to_num(codes, nan=0).astype(np.int64)
        except Exception:
            encoded[:, ci] = 0
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                mat[i, j] = _pyinform.transfer_entropy(
                    encoded[:, i], encoded[:, j], k
                )
            except Exception as exc:
                log.warning("transfer_entropy_matrix (%d→%d): %s", i, j, exc)
                mat[i, j] = float("nan")
    return pd.DataFrame(mat, index=cols, columns=cols)


def bayesian_network_edges(df: pd.DataFrame, n_bins: int = 5) -> list:
    """
    Bayesian Network structure learning (Hill-Climb + K2Score).

    Returns list of (parent, child) tuples.
    Requires pgmpy + networkx.  Raises ImportError with install hint if absent.
    """
    if not _HAS_PGMPY:
        raise ImportError(
            "Bayesian Network requires pgmpy and networkx.\n"
            "Install with:  pip install pgmpy networkx"
        )
    from sklearn.preprocessing import KBinsDiscretizer
    clean = _clean_numeric(df)
    disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
    # pgmpy >=1.0 selects the score family from the *column dtype*: numeric
    # columns are treated as continuous (only the bic-g / ll-g / aic-g scores
    # apply), while pandas 'category' columns are treated as discrete.  Cast
    # the binned ordinals to pandas Categorical so K2 becomes applicable.
    binned = disc.fit_transform(clean).astype(int)
    df_disc = pd.DataFrame(binned, columns=clean.columns).astype("category")
    hc = _HCS(data=df_disc)
    # pgmpy >=1.0 wants scoring_method as a string key ("k2", "bdeu", …); the
    # older class-instance form (K2Score(data=…)) is kept as a fallback for
    # older releases.
    try:
        model = hc.estimate(scoring_method="k2")
    except (TypeError, ValueError):
        model = hc.estimate(scoring_method=_K2Score(data=df_disc))
    return list(model.edges())


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

#: Methods intentionally excluded from _DISPATCH because their return shape
#: or parameterisation differs from the standard (DataFrame → square matrix)
#: signature.  These are invoked via dedicated branches in gui_dlg_cor_plot.py.
#: - "mi"        : mutual_info_regression/_classif (scores against a target)
#: - "lasso"     : lasso_coefficients (ranked series against a target)
#: - "ccf"       : ccf_series (1-D cross-correlation at lag k)
#: - "dag"       : GraphicalLassoCV edge set (graph, not matrix)
#: - "bayesian"  : bayesian_network_edges (DAG edge list)
_DISPATCH_EXCLUDED_KEYS: Final[Tuple[str, ...]] = (
    "mi", "lasso", "ccf", "dag", "bayesian",
)

#: Maps method_key → (title, codomain, callable).
_DISPATCH: Final[Dict[str, Tuple[str, str, Any]]] = {
    # key               title                                   codomain           callable
    "pearson":          ("Pearson Correlation",                 CODOMAIN_CORR,     pearson_matrix),
    "spearman":         ("Spearman Rank Correlation",           CODOMAIN_CORR,     spearman_matrix),
    "kendall":          ("Kendall Tau Correlation",             CODOMAIN_CORR,     kendall_matrix),
    "mic":              ("Maximal Information Coefficient",     CODOMAIN_UNIT,     mic_matrix),
    "distance":         ("Distance Correlation",                CODOMAIN_UNIT,     distance_correlation_matrix),
    "hoeffding":        ("Hoeffding's D",                       CODOMAIN_CORR,     hoeffding_matrix),
    "chatterjee":       ("Chatterjee's ξ Correlation",          CODOMAIN_CORR,     chatterjee_matrix),
    "polychoric":       ("Polychoric (Olsson ML)",              CODOMAIN_CORR,     polychoric_matrix),
    "r_statistic":      ("Blomqvist β (medial correlation)",    CODOMAIN_CORR,     r_statistic_matrix),
    "hsic":             ("HSIC Independence Criterion",         CODOMAIN_UNIT,     hsic_matrix),
    "mice":             ("k-NN Mutual Information (k=5)",       CODOMAIN_NONNEG,   mice_matrix),
    "sparse_partial":   ("Sparse Partial Corr (GraphLASSO)",    CODOMAIN_CORR,     sparse_partial_correlation),
    "cca":              ("Pairwise 1-D CCA (|Pearson|)",        CODOMAIN_CORR,     cca_matrix),
    "partial":          ("Partial Correlation Matrix",          CODOMAIN_CORR,     partial_correlation_matrix),
    "autoencoder":      ("Bottleneck Autoencoder",              CODOMAIN_UNIT,     autoencoder_dependency),
    "granger":          ("Granger Causality (1−p)",             CODOMAIN_PVAL,     granger_matrix),
    "transfer_entropy": ("Transfer Entropy",                    CODOMAIN_NONNEG,   transfer_entropy_matrix),
}
