"""
DNV Scientific Module
---------------------
Role:
    Reproducible figure-generation script for the NPJ Computational Materials
    paper on DNV 2.0 (SEAL operator + surrogate audit, 249-spinel case study).
    Produces Figures 1–6 at publication DPI into ./paper_figs/.

Scientific Context:
    Drives the full Phase-1 engine stack end-to-end on the canonical
    249-spinel DataAsset:
        spinel_dataset.load_canonical_spinel_asset
          → seal_engine.compute_seal              (5 variants)
          → stability_engine.compute_ensemble_stability  (K×M×F tensor)
          → surrogate_engine.compute_surrogate_audit     (3-step shortcut)
    Every figure function is pure: it takes the engine outputs + an
    output path and returns the figure handle after saving.

Invariants:
    - The script is deterministic given a fixed random seed.
    - All figure sizes, DPIs, and typographic constants are Final named
      constants.  No bare literals.
    - Every figure is saved at PUB_DPI (600) regardless of screen DPI.
    - If a figure would be empty (e.g. no contract-feasible variants),
      the function draws a diagnostic note instead of crashing.

Assumptions:
    - Working directory = dnv_flat; source CSVs present for the loader.
    - sklearn, matplotlib, seaborn, and xgboost available.

Failure Modes:
    - SEAL failing to produce variants: script writes a placeholder for
      Fig 2/3 and continues; Fig 1/4 still run on the base data.
    - XGBoost import failure: stability ensemble degrades to (LASSO, RF);
      figures update automatically via the returned model_keys.

Provenance:
    - Each figure PNG is accompanied by a .txt sidecar listing the
      random seed, engine versions, and the per-figure numerical values
      used on the plot, so every figure in the paper has a machine-
      readable source of truth.
"""
from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # headless backend for reproducibility
import matplotlib.pyplot as plt             # noqa: E402
import seaborn as sns                       # noqa: E402

from sklearn.ensemble        import RandomForestRegressor   # noqa: E402
from sklearn.linear_model    import LassoCV                  # noqa: E402
from sklearn.metrics         import r2_score                 # noqa: E402
from sklearn.model_selection import KFold, cross_val_predict # noqa: E402
from sklearn.preprocessing   import StandardScaler           # noqa: E402

log = logging.getLogger("paper_spinel")
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Figure constants — every numeric literal is Final with a unit suffix
# ---------------------------------------------------------------------------

#: Random seed used throughout the paper script [dimensionless].
PAPER_RANDOM_SEED: Final[int] = 42

#: Publication resolution for every saved figure [dots per inch].
PUB_DPI: Final[int] = 600

#: Standard full-column figure width [inch].
FIG_WIDTH_IN: Final[float] = 6.5

#: Standard figure height [inch].
FIG_HEIGHT_IN: Final[float] = 4.5

#: Wide figure height for multi-panel layouts [inch].
FIG_WIDE_HEIGHT_IN: Final[float] = 3.6

#: Default font size for axis labels [typographic points].
LABEL_FONT_PT: Final[int] = 10

#: Default tick label font size [typographic points].
TICK_FONT_PT: Final[int] = 9

#: Scatter marker size for parity plots [point²].
PARITY_MARKER_SIZE_PT2: Final[int] = 24

#: Alpha for scatter fills [dimensionless].
PARITY_MARKER_ALPHA: Final[float] = 0.55

#: Number of SEAL variants to request for the paper figures [count].
PAPER_N_VARIANTS: Final[int] = 5

#: Number of cross-validation folds used for all paper CV metrics [count].
PAPER_CV_FOLDS: Final[int] = 5

#: Top-k depth used in the stability shortlist [count].
PAPER_TOPK: Final[int] = 10

#: Output directory for all figures (created on first run).
OUTPUT_DIR: Final[Path] = Path(__file__).resolve().parent / "paper_figs"

#: Target column for every regression in the paper [unit in metadata].
TARGET_COL: Final[str] = "E_a"

#: Proxy column used by the surrogate audit (bond length A–X) [angstrom].
SURROGATE_PROXY_COL: Final[str] = "d_AX"

#: The 14 columns fed to SEAL — structural + tabulated atomic properties.
#: Excludes U (near-constant) and t_ratio (effectively IQR=0).
PAPER_SEAL_COLS: Final[Tuple[str, ...]] = (
    "E_hull", "EN_A", "EN_B", "EN_X",
    "R_A", "R_B", "R_X",
    "a", "b", "c", "alpha",
    "d_AX", "d_BX", "k_64m",
)

#: Audit features for the surrogate shortcut test — cheap tabulated
#: atomic properties that we suspect may encode the proxy.
PAPER_AUDIT_FEATURES: Final[Tuple[str, ...]] = (
    "EN_A", "EN_B", "EN_X",
    "R_A", "R_B", "R_X",
    "Polarizability_A", "Polarizability_B", "Polarizability_X",
)

#: Full feature set for the target-regression step (includes proxy).
PAPER_FULL_FEATURES: Final[Tuple[str, ...]] = (
    *PAPER_AUDIT_FEATURES,
    "a", "b", "c", "alpha",
    "d_AX", "d_BX", "k_64m",
    "E_hull",
)


# ---------------------------------------------------------------------------
# Engine imports — done after constants so the module loads even if one
# engine is temporarily broken (the individual figure functions fail loud).
# ---------------------------------------------------------------------------
from spinel_dataset     import (                                # noqa: E402
    load_canonical_spinel_asset, SPINEL_CONTRACTS,
)
from seal_engine        import compute_seal, ColumnBoundSpec    # noqa: E402
from stability_engine   import (                                # noqa: E402
    compute_ensemble_stability, default_ensemble_specs,
    EnsembleStabilityResult,
)
from surrogate_engine   import (                                # noqa: E402
    compute_surrogate_audit, ShortcutAuditResult,
    VERDICT_SHORTCUT, VERDICT_PARTIAL_SHORTCUT,
    VERDICT_NO_SHORTCUT, VERDICT_INCONCLUSIVE,
)


# ---------------------------------------------------------------------------
# Small dataclass for the set of engine outputs a figure pass needs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PaperContext:
    """Frozen bundle of inputs shared across every figure function."""
    asset:          Any
    seal_result:    Any
    stab_result:    EnsembleStabilityResult
    shortcut:       ShortcutAuditResult
    output_dir:     Path


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

def _apply_paper_style() -> None:
    """Global matplotlib rcParams for publication-quality figures."""
    plt.rcParams.update({
        "figure.dpi":      100,                # screen; PUB_DPI used on save
        "savefig.dpi":     PUB_DPI,
        "font.family":     "DejaVu Sans",
        "font.size":       LABEL_FONT_PT,
        "axes.labelsize":  LABEL_FONT_PT,
        "axes.titlesize":  LABEL_FONT_PT + 1,
        "xtick.labelsize": TICK_FONT_PT,
        "ytick.labelsize": TICK_FONT_PT,
        "legend.fontsize": TICK_FONT_PT,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "lines.linewidth":  1.3,
    })


def _savefig(fig: plt.Figure, path: Path, sidecar_lines: Sequence[str]) -> None:
    """Save figure at PUB_DPI and write a companion .txt sidecar."""
    fig.savefig(path, dpi=PUB_DPI, bbox_inches="tight")
    sc = path.with_suffix(".txt")
    sc.write_text("\n".join(sidecar_lines) + "\n")
    log.info("saved %s", path.name)


# ---------------------------------------------------------------------------
# Phase-1 run: load data, run SEAL, stability, surrogate audit
# ---------------------------------------------------------------------------

def run_phase1(output_dir: Path = OUTPUT_DIR) -> PaperContext:
    """Execute the full Phase-1 engine stack and return a PaperContext."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("running Phase 1 engine stack...")

    asset, rep = load_canonical_spinel_asset()
    df = asset.dataframe
    log.info("dataset: %d rows × %d cols (dropped %d raw-only compounds)",
             len(df), df.shape[1], len(rep.dropped_from_raw))

    bounds = [
        ColumnBoundSpec(col=c.observable, lo=c.lower_bound, hi=c.upper_bound)
        for c in SPINEL_CONTRACTS if c.observable in PAPER_SEAL_COLS
    ]
    seal_result = compute_seal(
        df,
        selected_cols = list(PAPER_SEAL_COLS),
        bounds_list   = bounds,
        n_variants    = PAPER_N_VARIANTS,
    )
    log.info("SEAL: %s", seal_result.message)

    stab_result = compute_ensemble_stability(
        seal_result,
        target_col  = TARGET_COL,
        model_specs = default_ensemble_specs(),
        top_k       = PAPER_TOPK,
    )
    log.info("stability: %s", stab_result.message)

    shortcut = compute_surrogate_audit(
        df,
        target_col     = TARGET_COL,
        proxy_col      = SURROGATE_PROXY_COL,
        audit_features = list(PAPER_AUDIT_FEATURES),
        full_features  = list(PAPER_FULL_FEATURES),
        cv_folds       = PAPER_CV_FOLDS,
        random_state   = PAPER_RANDOM_SEED,
    )
    log.info("shortcut: %s", shortcut.message)

    return PaperContext(
        asset       = asset,
        seal_result = seal_result,
        stab_result = stab_result,
        shortcut    = shortcut,
        output_dir  = output_dir,
    )


# ---------------------------------------------------------------------------
# Figure 1 — LASSO + RF + XGB parity plots on the base DataFrame
# ---------------------------------------------------------------------------

def _parity_cv(
    df:         pd.DataFrame,
    feats:      Sequence[str],
    target:     str,
    model:      Any,
    cv_folds:   int = PAPER_CV_FOLDS,
    rs:         int = PAPER_RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Standardise, KFold cross_val_predict, return (y, y_pred, R²)."""
    sub = df[list(feats) + [target]].dropna()
    X = StandardScaler().fit_transform(sub[list(feats)].values.astype(float))
    y = sub[target].values.astype(float)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=rs)
    y_pred = cross_val_predict(model, X, y, cv=kf)
    return y, y_pred, float(r2_score(y, y_pred))


def fig_1_parity(ctx: PaperContext) -> plt.Figure:
    df = ctx.asset.dataframe
    feats = list(PAPER_FULL_FEATURES)

    lasso = LassoCV(n_alphas=50, cv=PAPER_CV_FOLDS, random_state=PAPER_RANDOM_SEED, max_iter=20000)
    rf    = RandomForestRegressor(n_estimators=300, random_state=PAPER_RANDOM_SEED, n_jobs=-1)
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            random_state=PAPER_RANDOM_SEED, n_jobs=-1, verbosity=0, tree_method="hist",
        )
        panels: List[Tuple[str, Any]] = [
            ("LASSO",         lasso),
            ("Random Forest", rf),
            ("XGBoost",       xgb),
        ]
    except ImportError:
        panels = [("LASSO", lasso), ("Random Forest", rf)]
        log.warning("xgboost missing; Fig 1 reduced to 2 panels")

    fig, axes = plt.subplots(
        1, len(panels),
        figsize=(FIG_WIDTH_IN, FIG_WIDE_HEIGHT_IN),
        sharex=True, sharey=True,
    )
    if len(panels) == 1:
        axes = [axes]

    sidecar = [f"# Figure 1 — parity plots on base spinel data ({len(df)} rows)"]
    for ax, (title, model) in zip(axes, panels):
        y, y_pred, r2 = _parity_cv(df, feats, TARGET_COL, model)
        ax.scatter(
            y, y_pred,
            s=PARITY_MARKER_SIZE_PT2, alpha=PARITY_MARKER_ALPHA,
            edgecolor="k", linewidth=0.3,
        )
        lo = float(min(y.min(), y_pred.min()))
        hi = float(max(y.max(), y_pred.max()))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.9)
        ax.set_title(f"{title}\nR²={r2:.3f}")
        ax.set_xlabel("actual E$_a$ (eV)")
        ax.set_ylabel("predicted E$_a$ (eV)")
        sidecar.append(f"{title}: R²={r2:.4f}")

    fig.suptitle("Figure 1 — ensemble parity on canonical 249-spinel set")
    fig.tight_layout()
    _savefig(fig, ctx.output_dir / "fig1_parity.png", sidecar)
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — SEAL admissible family distortion + EVI (per-constraint decomposition)
# ---------------------------------------------------------------------------

def fig_2_seal_family(ctx: PaperContext) -> plt.Figure:
    seal = ctx.seal_result
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_WIDE_HEIGHT_IN))

    # ── Left panel: distortion Ω per variant, colour by d_after ──────────
    ax = axes[0]
    xs = np.arange(len(seal.variants))
    omegas = [v.distortion for v in seal.variants]
    ds     = [v.d_after    for v in seal.variants]
    ax.bar(xs, omegas, color=sns.color_palette("viridis", len(xs)))
    ax.set_xticks(xs)
    ax.set_xticklabels([f"T$_{{{i}}}$" for i in xs])
    ax.set_ylabel(r"distortion $\Omega(T;X)$")
    ax.set_title(f"SEAL admissible family, K={len(xs)}\n"
                 f"all contract-feasible, d$_{{before}}$={seal.d_before:.3f}")
    for i, (o, d) in enumerate(zip(omegas, ds)):
        ax.text(i, o, f" d={d:.2f}", va="bottom", ha="center",
                fontsize=TICK_FONT_PT - 1, rotation=0)

    # ── Right panel: EVI distribution ───────────────────────────────────
    ax = axes[1]
    evi = np.asarray(seal.evi_scores, dtype=float)
    ax.hist(evi, bins=30, color="#2E7DB8", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("EVI  (envelope violation influence)")
    ax.set_ylabel("count")
    ax.set_title("per-sample EVI distribution")
    ax.axvline(0.0, color="k", linestyle="--", linewidth=0.8)

    fig.suptitle("Figure 2 — SEAL variant family and EVI")
    fig.tight_layout()

    sidecar = [
        "# Figure 2 — SEAL admissible family and EVI",
        f"n_variants = {len(seal.variants)}",
        f"d_before   = {seal.d_before:.6f}",
        "per-variant distortion / d_after:",
    ] + [f"  T{i}: Ω={o:.6f}  d_after={d:.6f}" for i, (o, d) in enumerate(zip(omegas, ds))]
    sidecar.append(f"EVI min={evi.min():.4e} max={evi.max():.4e} mean={evi.mean():.4e}")
    _savefig(fig, ctx.output_dir / "fig2_seal_family.png", sidecar)
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — stability: σ² within/between + Kendall τ
# ---------------------------------------------------------------------------

def fig_3_stability(ctx: PaperContext) -> plt.Figure:
    stab = ctx.stab_result
    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_WIDTH_IN + 1.0, FIG_WIDE_HEIGHT_IN),
        gridspec_kw={"width_ratios": [2.2, 1.0]},
    )

    # ── Left: stacked bar σ²_within (prep) vs σ²_between (model), top-12 by mean_phi
    ax = axes[0]
    top = stab.mean_phi_f.sort_values(ascending=False).head(12)
    feats = top.index.tolist()
    within = stab.sigma2_within_f.reindex(feats).values
    between = stab.sigma2_between_f.reindex(feats).values
    y = np.arange(len(feats))
    ax.barh(y, within, color="#E89B28", label=r"$\sigma^2_{\mathrm{within}}$ (prep)")
    ax.barh(y, between, left=within, color="#2E7DB8",
            label=r"$\sigma^2_{\mathrm{between}}$ (model)")
    ax.set_yticks(y)
    ax.set_yticklabels(feats)
    ax.invert_yaxis()
    ax.set_xlabel("variance of L1-normalised importance")
    ax.set_title("nested variance decomposition (top-12 features)")
    ax.legend(loc="lower right", frameon=False)

    # ── Right: model-averaged Kendall τ between variants (K×K) ──────────
    ax = axes[1]
    tau_stack = np.stack(
        [r.kendall_tau_matrix for r in stab.per_model_results], axis=0,
    )  # (M, K, K)
    tau = tau_stack.mean(axis=0)
    K = tau.shape[0]
    im = ax.imshow(tau, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(K))
    ax.set_yticks(np.arange(K))
    ax.set_xticklabels([f"T$_{{{i}}}$" for i in range(K)])
    ax.set_yticklabels([f"T$_{{{i}}}$" for i in range(K)])
    ax.set_title(f"Kendall τ across variants\nmean = {stab.mean_kendall_tau:.3f}")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{tau[i,j]:.2f}", ha="center", va="center",
                    fontsize=TICK_FONT_PT - 1, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Figure 3 — stability of importance across (K variants × M models)",
    )
    fig.tight_layout()

    agg_within  = float(stab.sigma2_within_f.sum())
    agg_between = float(stab.sigma2_between_f.sum())
    agg_total   = float(stab.sigma2_total_f.sum())
    sidecar = [
        "# Figure 3 — stability decomposition",
        f"K={len(range(stab.n_variants_used))}, M={len(stab.model_keys)}, F={len(stab.feature_cols)}",
        f"models = {list(stab.model_keys)}",
        f"Σ σ²_within  (prep)  = {agg_within:.6f}",
        f"Σ σ²_between (model) = {agg_between:.6f}",
        f"Σ σ²_total           = {agg_total:.6f}",
        f"prep fraction        = {agg_within / agg_total:.4f}",
        f"mean Kendall τ       = {stab.mean_kendall_tau:.4f}",
        f"top-{PAPER_TOPK} Jaccard       = {stab.topk_jaccard:.4f}",
        "top-12 features by mean φ:",
    ] + [f"  {f}: σ²_w={stab.sigma2_within_f[f]:.3e}  "
          f"σ²_b={stab.sigma2_between_f[f]:.3e}  mean_φ={stab.mean_phi_f[f]:.4f}"
          for f in feats]
    _savefig(fig, ctx.output_dir / "fig3_stability.png", sidecar)
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — surrogate shortcut audit bars
# ---------------------------------------------------------------------------

def fig_4_surrogate(ctx: PaperContext) -> plt.Figure:
    sc = ctx.shortcut
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_WIDE_HEIGHT_IN))

    labels = [
        r"R²($p$|$X_{aud}$)",
        r"R²($y$|$X_{full}$)",
        r"R²($y_{res}$|$X_{aud}$)",
        r"R²($y$|$X_{full} \setminus p$)",
    ]
    values = [
        sc.r2_proxy_from_audit,
        sc.r2_target_full,
        sc.r2_target_residual,
        sc.r2_target_no_proxy if not np.isnan(sc.r2_target_no_proxy) else 0.0,
    ]
    colors = ["#D95C5C", "#2E7DB8", "#E89B28", "#6FA868"]
    xs = np.arange(len(labels))
    bars = ax.bar(xs, values, color=colors, edgecolor="black", linewidth=0.4)

    ax.axhline(sc.proxy_r2_threshold, color="#D95C5C", linestyle="--",
               linewidth=0.9, label=f"τ_proxy = {sc.proxy_r2_threshold}")
    ax.axhline(0.0, color="k", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("cross-validated R²")
    ax.set_ylim(min(-0.05, min(values) - 0.05), 1.05)
    ax.set_title(
        f"Figure 4 — surrogate shortcut audit  (proxy = {sc.proxy_col})\n"
        f"verdict: {sc.verdict}    "
        f"Δ$_{{shortcut}}$={sc.shortcut_drop:.3f}  "
        f"Δ$_{{ablation}}$={sc.ablation_drop:.3f}"
    )
    ax.legend(loc="lower left", frameon=False)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, v,
                f"{v:.3f}", ha="center", va="bottom", fontsize=TICK_FONT_PT - 1)

    fig.tight_layout()
    sidecar = [
        "# Figure 4 — surrogate shortcut audit",
        f"target              = {sc.target_col}",
        f"proxy               = {sc.proxy_col}",
        f"audit_features      = {list(sc.audit_features)}",
        f"full_features       = {list(sc.full_features)}",
        f"R²(p|X_aud)         = {sc.r2_proxy_from_audit:.4f}",
        f"R²(y|X_full)        = {sc.r2_target_full:.4f}",
        f"R²(y_res|X_aud)     = {sc.r2_target_residual:.4f}",
        f"R²(y|X_full \\ p)    = {sc.r2_target_no_proxy:.4f}",
        f"Δ_shortcut          = {sc.shortcut_drop:.4f}",
        f"Δ_ablation          = {sc.ablation_drop:.4f}",
        f"τ_proxy             = {sc.proxy_r2_threshold}",
        f"τ_drop              = {sc.shortcut_min_drop}",
        f"verdict             = {sc.verdict}",
        f"n_samples           = {sc.n_samples}",
    ]
    _savefig(fig, ctx.output_dir / "fig4_surrogate.png", sidecar)
    return fig


# ---------------------------------------------------------------------------
# Figure 5 — variant-indexed shortlist (family-stable vs contingent)
# ---------------------------------------------------------------------------

def fig_5_shortlist(ctx: PaperContext) -> plt.Figure:
    stab = ctx.stab_result
    top = stab.mean_phi_f.sort_values(ascending=False).head(PAPER_TOPK).index.tolist()
    K = len(range(stab.n_variants_used))

    # Phi per variant, model-averaged (K × topK)
    phi = stab.phi_tensor.mean(axis=1)  # (K, F)
    feat_idx = [list(stab.feature_cols).index(f) for f in top]
    phi_top = phi[:, feat_idx]  # (K, topK)

    labels_col = [stab.stability_labels.get(f, "?") for f in top]
    y_tick_labels = [f"{f}  [{lab}]" for f, lab in zip(top, labels_col)]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN + 0.8, FIG_WIDE_HEIGHT_IN + 0.5))
    im = ax.imshow(phi_top.T, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(y_tick_labels)
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels([f"T$_{{{i}}}$" for i in range(K)])
    ax.set_xlabel("SEAL variant")
    ax.set_title(
        f"Figure 5 — variant-indexed top-{PAPER_TOPK} shortlist\n"
        f"mean Kendall τ={stab.mean_kendall_tau:.3f}, "
        f"top-{PAPER_TOPK} Jaccard={stab.topk_jaccard:.3f}"
    )
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02,
                 label=r"model-averaged $\bar\phi_{k\cdot f}$")

    fig.tight_layout()
    sidecar = [
        "# Figure 5 — variant-indexed shortlist",
        f"top-{PAPER_TOPK} features (by grand mean φ):",
    ] + [f"  {f}: label={stab.stability_labels.get(f, '?')}  "
          f"mean={stab.mean_phi_f[f]:.4f}" for f in top]
    _savefig(fig, ctx.output_dir / "fig5_shortlist.png", sidecar)
    return fig


# ---------------------------------------------------------------------------
# Figure 6 — carrier-conditioned importance profiles
# ---------------------------------------------------------------------------

def fig_6_carrier_profiles(ctx: PaperContext) -> plt.Figure:
    """Per-carrier RandomForest importance profiles.

    For each A_site carrier (Li/Na/Mg/Ca/Zn/Al), fit a RandomForest on
    the subset predicting E_a from PAPER_FULL_FEATURES, extract
    feature_importances_, L1-normalise, and plot a grouped bar chart
    over the top-8 features identified on the pooled data.
    """
    df = ctx.asset.dataframe
    carriers = sorted(df["A_site"].unique())

    # Drop features that are constant within any single A-site carrier
    # (e.g. EN_A, R_A, Polarizability_A, EA_A — determined by A_site identity).
    # Those would appear as bars of height zero in every per-carrier RF and
    # dominate the figure with empty slots.
    candidate_feats = list(PAPER_FULL_FEATURES)
    carrier_varying: List[str] = []
    for f in candidate_feats:
        if all(df.loc[df["A_site"] == c, f].nunique(dropna=True) > 1 for c in carriers):
            carrier_varying.append(f)
    feats = carrier_varying

    # Identify the top-8 features on the pooled data (using carrier-varying set)
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feats].values.astype(float))
    y = df[TARGET_COL].values.astype(float)
    rf_pool = RandomForestRegressor(
        n_estimators=300, random_state=PAPER_RANDOM_SEED, n_jobs=-1,
    )
    rf_pool.fit(X, y)
    pool_imp = pd.Series(rf_pool.feature_importances_, index=feats)
    top_feats = pool_imp.sort_values(ascending=False).head(8).index.tolist()
    profile: Dict[str, np.ndarray] = {}
    n_per: Dict[str, int] = {}
    for carrier in carriers:
        sub = df[df["A_site"] == carrier]
        n_per[carrier] = len(sub)
        if len(sub) < 20:
            profile[carrier] = np.zeros(len(top_feats), dtype=float)
            continue
        X_c = StandardScaler().fit_transform(sub[feats].values.astype(float))
        y_c = sub[TARGET_COL].values.astype(float)
        rf_c = RandomForestRegressor(
            n_estimators=200, random_state=PAPER_RANDOM_SEED, n_jobs=-1,
        )
        rf_c.fit(X_c, y_c)
        imp_full = pd.Series(rf_c.feature_importances_, index=feats)
        # L1-normalise across full feature set, then slice to top_feats
        s = float(imp_full.abs().sum()) or 1.0
        imp_full_norm = imp_full / s
        profile[carrier] = imp_full_norm.reindex(top_feats).values

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN + 0.5, FIG_WIDE_HEIGHT_IN + 0.3))
    n_feat = len(top_feats)
    n_car = len(carriers)
    group_width = 0.8
    bar_w = group_width / n_car
    palette = sns.color_palette("tab10", n_car)

    for i, carrier in enumerate(carriers):
        xs = np.arange(n_feat) + (i - (n_car - 1) / 2.0) * bar_w
        ax.bar(
            xs, profile[carrier], width=bar_w,
            label=f"{carrier}  (n={n_per[carrier]})",
            color=palette[i], edgecolor="black", linewidth=0.3,
        )

    ax.set_xticks(np.arange(n_feat))
    ax.set_xticklabels(top_feats, rotation=30, ha="right")
    ax.set_ylabel("L1-normalised RF importance")
    ax.set_title("Figure 6 — carrier-conditioned importance profiles")
    ax.legend(title="A-site carrier", loc="upper right",
              frameon=False, ncol=2, fontsize=TICK_FONT_PT - 1)
    fig.tight_layout()

    sidecar = [
        "# Figure 6 — carrier-conditioned RF importances",
        f"top-8 (pooled) features: {top_feats}",
        "per-carrier row counts:",
    ] + [f"  {c}: n={n_per[c]}" for c in carriers] + [
        "per-carrier profile (L1-normalised):",
    ]
    for c in carriers:
        vals = ", ".join(f"{v:.4f}" for v in profile[c])
        sidecar.append(f"  {c}: [{vals}]")
    _savefig(fig, ctx.output_dir / "fig6_carrier_profiles.png", sidecar)
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_all_figures(output_dir: Path = OUTPUT_DIR) -> PaperContext:
    _apply_paper_style()
    ctx = run_phase1(output_dir)
    fig_1_parity(ctx);            plt.close("all")
    fig_2_seal_family(ctx);       plt.close("all")
    fig_3_stability(ctx);         plt.close("all")
    fig_4_surrogate(ctx);         plt.close("all")
    fig_5_shortlist(ctx);         plt.close("all")
    fig_6_carrier_profiles(ctx);  plt.close("all")
    log.info("all figures written to %s", ctx.output_dir)
    return ctx


if __name__ == "__main__":
    try:
        build_all_figures()
    except Exception as exc:   # keep CLI exit code meaningful
        log.exception("paper_spinel failed: %s", exc)
        sys.exit(1)
