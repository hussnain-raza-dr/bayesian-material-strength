"""
visualization.py
================

Reusable, publication-quality plotting functions for Bayesian strength
model analysis.

All functions return ``(fig, ax)`` tuples for composability and accept an
optional ``save_path`` argument for writing 300 DPI PNG files to ``results/``.

Plotting style
--------------
Figures use a minimal academic style (seaborn-v0_8-whitegrid base) with
consistent colour maps:
    - Posterior distributions: viridis / steel-blue
    - Credible intervals: shaded alpha layers (50% and 90%)
    - Group comparisons: tab10 categorical palette
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Global style settings ─────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
mpl.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.constrained_layout.use": True,
    }
)

_PALETTE = sns.color_palette("tab10")
_STEEL_BLUE = "#2a6496"
_ORANGE = "#d35400"
_GREEN = "#27ae60"


def _save(fig: plt.Figure, save_path: Optional[Union[str, Path]]) -> None:
    """Save figure if save_path is provided."""
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Posterior distribution plots
# ──────────────────────────────────────────────────────────────────────────────


def plot_posterior(
    idata: az.InferenceData,
    var_names: list[str],
    true_values: Optional[dict[str, float]] = None,
    title: str = "Posterior Distributions",
    save_path: Optional[Union[str, Path]] = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot marginal posterior distributions with optional ground-truth lines.

    Uses ArviZ's ``plot_posterior`` internally but wraps it for consistent
    styling and ``results/`` saving.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData with posterior group.
    var_names : list[str]
        Variables to plot (must match keys in ``idata.posterior``).
    true_values : dict[str, float] or None, optional
        Ground-truth parameter values (if known from data generation) to
        overlay as vertical lines. Keys must match ``var_names``.
    title : str, optional
        Figure suptitle. Default is ``"Posterior Distributions"``.
    save_path : str or Path or None, optional
        Output path. If None, figure is not saved.

    Returns
    -------
    fig : plt.Figure
    axes : np.ndarray of plt.Axes
    """
    # Map var_names to actual variable names in idata, handling model prefixes
    actual_var_names = []
    var_name_mapping = {}  # Map from short name to actual name in idata
    
    for var in var_names:
        # Try to find the variable in posterior
        if var in idata.posterior:
            actual_var_names.append(var)
            var_name_mapping[var] = var
        else:
            # Try with common prefixes
            found = False
            for prefix in ["bayesian_lr", "hierarchical"]:
                full_name = f"{prefix}::{var}"
                if full_name in idata.posterior:
                    actual_var_names.append(full_name)
                    var_name_mapping[var] = full_name
                    found = True
                    break
            if not found:
                # Last resort: find any variable containing the var_name
                matching = [v for v in idata.posterior.data_vars if var in v and "mu_dim" not in v]
                if matching:
                    actual_var_names.append(matching[0])
                    var_name_mapping[var] = matching[0]
                else:
                    # If still not found, just use the original name and let ArviZ handle the error
                    actual_var_names.append(var)
                    var_name_mapping[var] = var
    
    axes = az.plot_posterior(
        idata,
        var_names=actual_var_names,
        ref_val=None,
        figsize=(4 * len(actual_var_names), 4),
        textsize=11,
        round_to=3,
        kind="kde",
        color=_STEEL_BLUE,
    )
    fig = plt.gcf()
    
    # Overlay ground-truth reference lines if provided
    if true_values:
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        for i, var_name in enumerate(var_names):
            if var_name in true_values and i < len(axes_flat):
                ref_val = true_values[var_name]
                axes_flat[i].axvline(
                    ref_val, 
                    color='red', 
                    linestyle='--', 
                    linewidth=2, 
                    label=f'True value: {ref_val:.3g}',
                    alpha=0.7
                )
                axes_flat[i].legend(loc='upper right')
    
    fig.suptitle(title, fontsize=14, y=1.02)
    _save(fig, save_path)
    return fig, axes


# ──────────────────────────────────────────────────────────────────────────────
# 2. Posterior predictive intervals
# ──────────────────────────────────────────────────────────────────────────────


def plot_predictive_intervals(
    temperature_grid: np.ndarray,
    predictive_samples: np.ndarray,
    observed_temperature: np.ndarray,
    observed_strength: np.ndarray,
    carbon_ref: float = 0.30,
    title: str = "Posterior Predictive Distribution",
    save_path: Optional[Union[str, Path]] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot 50% and 90% posterior predictive credible intervals vs. temperature.

    Parameters
    ----------
    temperature_grid : np.ndarray, shape (n_grid,)
        Temperature values at which predictions were made.
    predictive_samples : np.ndarray, shape (n_samples, n_grid)
        Posterior predictive samples from ``model.predict()``.
    observed_temperature : np.ndarray, shape (n_obs,)
        Observed temperature values for scatter overlay.
    observed_strength : np.ndarray, shape (n_obs,)
        Observed tensile strength values (MPa).
    carbon_ref : float, optional
        Reference carbon content used in predictions (for axis label).
        Default is 0.30 wt%.
    title : str, optional
        Axis title. Default is ``"Posterior Predictive Distribution"``.
    save_path : str or Path or None, optional
        Output path for figure saving.

    Returns
    -------
    fig : plt.Figure
    ax : plt.Axes
    """
    median = np.percentile(predictive_samples, 50, axis=0)
    lo50, hi50 = np.percentile(predictive_samples, [25, 75], axis=0)
    lo90, hi90 = np.percentile(predictive_samples, [5, 95], axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))

    # 90% credible interval
    ax.fill_between(
        temperature_grid, lo90, hi90,
        alpha=0.25, color=_STEEL_BLUE, label="90% CI"
    )
    # 50% credible interval
    ax.fill_between(
        temperature_grid, lo50, hi50,
        alpha=0.45, color=_STEEL_BLUE, label="50% CI"
    )
    # Posterior median
    ax.plot(temperature_grid, median, color=_STEEL_BLUE, lw=2, label="Posterior median")

    # Observed data
    ax.scatter(
        observed_temperature, observed_strength,
        s=18, alpha=0.5, color="black", zorder=5, label="Observed"
    )

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Tensile Strength (MPa)")
    ax.set_title(f"{title}\n(C = {carbon_ref:.2f} wt%)")
    ax.legend(loc="upper right")
    sns.despine(fig=fig)
    _save(fig, save_path)
    return fig, ax


# ──────────────────────────────────────────────────────────────────────────────
# 3. Group-level comparison (hierarchical model)
# ──────────────────────────────────────────────────────────────────────────────


def plot_group_comparison(
    idata_hierarchical: az.InferenceData,
    idata_no_pool: Optional[list[az.InferenceData]] = None,
    group_labels: Optional[list[str]] = None,
    true_alphas: Optional[dict[str, float]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Shrinkage plot: partial-pooling vs. no-pooling group intercepts.

    Visualises the hallmark of hierarchical models: estimates for groups with
    fewer observations are pulled (shrunk) toward the global mean, trading a
    small bias for reduced variance.

    Parameters
    ----------
    idata_hierarchical : az.InferenceData
        Posterior from the hierarchical model (partial pooling).
    idata_no_pool : list[az.InferenceData] or None, optional
        List of separate single-group posteriors (no pooling).
        If None, only partial-pooling estimates are shown.
    group_labels : list[str] or None, optional
        Group names for axis labels. Default is the three alloy families.
    true_alphas : dict[str, float] or None, optional
        Ground-truth intercepts for reference lines.
    save_path : str or Path or None, optional
        Output path.

    Returns
    -------
    fig : plt.Figure
    ax : plt.Axes
    """
    if group_labels is None:
        group_labels = ["Austenitic", "Ferritic", "Martensitic"]

    n_groups = len(group_labels)
    y_pos = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(9, 5))

    # ── Partial-pooling estimates ─────────────────────────────────────────────
    alpha_pp = None
    for var_name in ["hierarchical::alpha", "alpha", "bayesian_lr_austenitic::alpha"]:
        try:
            alpha_pp = idata_hierarchical.posterior[var_name].values
            break
        except KeyError:
            continue
    
    if alpha_pp is None:
        # Last resort: find any variable ending with 'alpha' that has alloy dimension
        alpha_vars = [v for v in idata_hierarchical.posterior.data_vars if 'alpha' in v and 'mu' not in v]
        if alpha_vars:
            alpha_pp = idata_hierarchical.posterior[alpha_vars[0]].values
        else:
            raise KeyError(f"Cannot find 'alpha' variable in posterior. Available: {list(idata_hierarchical.posterior.data_vars)}")

    alpha_pp_flat = alpha_pp.reshape(-1, n_groups)
    means_pp = alpha_pp_flat.mean(axis=0)
    hdi_pp = az.hdi(alpha_pp_flat, hdi_prob=0.94)

    for i, (lbl, mean, hdi) in enumerate(zip(group_labels, means_pp, hdi_pp)):
        ax.errorbar(
            mean, y_pos[i],
            xerr=[[mean - hdi[0]], [hdi[1] - mean]],
            fmt="o", color=_STEEL_BLUE, ms=8, lw=2, capsize=4,
            label="Partial pooling" if i == 0 else "_nolegend_",
        )

    # ── No-pooling estimates ─────────────────────────────────────────────────
    if idata_no_pool is not None:
        for i, (idata_np, lbl) in enumerate(zip(idata_no_pool, group_labels)):
            # Try multiple possible variable name formats
            a_np = None
            for var_name in ["bayesian_lr::alpha", "alpha", "bayesian_lr_austenitic::alpha"]:
                try:
                    a_np = idata_np.posterior[var_name].values.flatten()
                    break
                except KeyError:
                    continue
            
            if a_np is None:
                # Last resort: find any variable ending with 'alpha'
                alpha_vars = [v for v in idata_np.posterior.data_vars if 'alpha' in v]
                if alpha_vars:
                    a_np = idata_np.posterior[alpha_vars[0]].values.flatten()
                else:
                    continue  # Skip this group if no alpha found
            
            mean_np = a_np.mean()
            hdi_np = az.hdi(a_np, hdi_prob=0.94)
            ax.errorbar(
                mean_np, y_pos[i] + 0.2,
                xerr=[[mean_np - hdi_np[0]], [hdi_np[1] - mean_np]],
                fmt="s", color=_ORANGE, ms=8, lw=2, capsize=4, alpha=0.8,
                label="No pooling" if i == 0 else "_nolegend_",
            )

    # ── True values ───────────────────────────────────────────────────────────
    if true_alphas is not None:
        for i, lbl in enumerate(group_labels):
            key = lbl.lower()
            if key in true_alphas:
                ax.axvline(
                    true_alphas[key], color=_GREEN, lw=1.5, ls="--",
                    label="True value" if i == 0 else "_nolegend_",
                    alpha=0.8,
                )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(group_labels)
    ax.set_xlabel("Group Intercept α (MPa)")
    ax.set_title("Partial Pooling vs. No Pooling: Group Intercepts\n(94% HDI)")
    ax.legend(loc="lower right")
    sns.despine(fig=fig)
    _save(fig, save_path)
    return fig, ax


# ──────────────────────────────────────────────────────────────────────────────
# 4. Tractability comparison (NUTS vs. ADVI)
# ──────────────────────────────────────────────────────────────────────────────


def plot_tractability_comparison(
    idata_nuts: az.InferenceData,
    idata_advi: az.InferenceData,
    var_names: list[str],
    time_nuts: float,
    time_advi: float,
    save_path: Optional[Union[str, Path]] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Side-by-side posterior mean and SD comparison: NUTS vs. ADVI.

    Parameters
    ----------
    idata_nuts : az.InferenceData
        Posterior from NUTS sampling (reference).
    idata_advi : az.InferenceData
        Posterior from ADVI variational inference.
    var_names : list[str]
        Scalar variables to compare (must exist in both posteriors).
    time_nuts : float
        Wall-clock sampling time for NUTS (seconds).
    time_advi : float
        Wall-clock optimisation time for ADVI (seconds).
    save_path : str or Path or None, optional
        Output path.

    Returns
    -------
    fig : plt.Figure
    axes : np.ndarray of plt.Axes  (shape: (2,))
    """
    nuts_means, nuts_sds = [], []
    advi_means, advi_sds = [], []

    for v in var_names:
        for idata, means_list, sds_list in [
            (idata_nuts, nuts_means, nuts_sds),
            (idata_advi, advi_means, advi_sds),
        ]:
            # Try with model name prefix
            for key in [f"bayesian_lr::{v}", v]:
                if key in idata.posterior:
                    vals = idata.posterior[key].values.flatten()
                    means_list.append(float(vals.mean()))
                    sds_list.append(float(vals.std()))
                    break

    x = np.arange(len(var_names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # ── Panel A: Posterior means ──────────────────────────────────────────────
    axes[0].bar(x - width / 2, nuts_means, width, label=f"NUTS ({time_nuts:.1f}s)",
                color=_STEEL_BLUE, alpha=0.85, edgecolor="white")
    axes[0].bar(x + width / 2, advi_means, width, label=f"ADVI ({time_advi:.1f}s)",
                color=_ORANGE, alpha=0.85, edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(var_names, rotation=20)
    axes[0].set_ylabel("Posterior Mean")
    axes[0].set_title("Posterior Means: NUTS vs. ADVI")
    axes[0].legend()

    # ── Panel B: Posterior standard deviations ───────────────────────────────
    axes[1].bar(x - width / 2, nuts_sds, width, label=f"NUTS ({time_nuts:.1f}s)",
                color=_STEEL_BLUE, alpha=0.85, edgecolor="white")
    axes[1].bar(x + width / 2, advi_sds, width, label=f"ADVI ({time_advi:.1f}s)",
                color=_ORANGE, alpha=0.85, edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(var_names, rotation=20)
    axes[1].set_ylabel("Posterior SD")
    axes[1].set_title("Posterior Uncertainty: NUTS vs. ADVI")
    axes[1].legend()

    fig.suptitle(
        f"Tractability Comparison — NUTS ({time_nuts:.1f}s) vs. ADVI ({time_advi:.1f}s)",
        fontsize=13,
    )
    sns.despine(fig=fig)
    _save(fig, save_path)
    return fig, axes


# ──────────────────────────────────────────────────────────────────────────────
# 5. Failure probability curve
# ──────────────────────────────────────────────────────────────────────────────


def plot_failure_probability(
    temperature_grid: np.ndarray,
    predictive_samples: np.ndarray,
    threshold_mpa: float = 400.0,
    carbon_ref: float = 0.30,
    save_path: Optional[Union[str, Path]] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot P(strength < threshold) as a function of temperature.

    Computes the empirical failure probability at each grid point from
    the posterior predictive samples. This quantity is directly usable in
    structural reliability analysis (Level II / Level III methods).

    Parameters
    ----------
    temperature_grid : np.ndarray, shape (n_grid,)
        Temperature values in °C.
    predictive_samples : np.ndarray, shape (n_samples, n_grid)
        Posterior predictive samples from ``model.predict()``.
    threshold_mpa : float, optional
        Strength threshold for failure definition (MPa). Default is 400.
    carbon_ref : float, optional
        Reference carbon content for axis annotation. Default is 0.30 wt%.
    save_path : str or Path or None, optional
        Output path.

    Returns
    -------
    fig : plt.Figure
    ax : plt.Axes
    """
    p_fail = (predictive_samples < threshold_mpa).mean(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(temperature_grid, p_fail, color=_ORANGE, lw=2.5)
    ax.fill_between(temperature_grid, 0, p_fail, alpha=0.2, color=_ORANGE)
    ax.axhline(0.05, color="gray", ls="--", lw=1.2, label="5% threshold")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(f"P(σ_y < {threshold_mpa:.0f} MPa)")
    ax.set_title(
        f"Failure Probability vs. Temperature\n"
        f"(C = {carbon_ref:.2f} wt%, threshold = {threshold_mpa:.0f} MPa)"
    )
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    sns.despine(fig=fig)
    _save(fig, save_path)
    return fig, ax


# ──────────────────────────────────────────────────────────────────────────────
# 6. Calibration curve
# ──────────────────────────────────────────────────────────────────────────────


def plot_calibration(
    observed: np.ndarray,
    predictive_samples: np.ndarray,
    n_levels: int = 20,
    save_path: Optional[Union[str, Path]] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot empirical vs. nominal credible interval coverage (calibration).

    A well-calibrated Bayesian model should produce coverage curves that
    lie on the diagonal: the 80% CI should contain 80% of test observations.
    Systematic departure indicates model misspecification or overconfidence.

    Parameters
    ----------
    observed : np.ndarray, shape (n_obs,)
        Held-out observed tensile strength values.
    predictive_samples : np.ndarray, shape (n_samples, n_obs)
        Posterior predictive samples at the held-out covariate values.
    n_levels : int, optional
        Number of credible level points to evaluate. Default is 20.
    save_path : str or Path or None, optional
        Output path.

    Returns
    -------
    fig : plt.Figure
    ax : plt.Axes
    """
    nominal_levels = np.linspace(0.0, 1.0, n_levels + 2)[1:-1]
    empirical_coverage = []

    for level in nominal_levels:
        alpha_half = (1 - level) / 2
        lo = np.percentile(predictive_samples, 100 * alpha_half, axis=0)
        hi = np.percentile(predictive_samples, 100 * (1 - alpha_half), axis=0)
        coverage = float(np.mean((observed >= lo) & (observed <= hi)))
        empirical_coverage.append(coverage)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Ideal calibration")
    ax.plot(nominal_levels, empirical_coverage, "o-", color=_STEEL_BLUE,
            lw=2, ms=5, label="Model calibration")
    ax.fill_between(
        nominal_levels,
        [n - 0.05 for n in nominal_levels],
        [n + 0.05 for n in nominal_levels],
        alpha=0.12, color="gray", label="±5% band"
    )
    ax.set_xlabel("Nominal Credible Level")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("Posterior Predictive Calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    sns.despine(fig=fig)
    _save(fig, save_path)
    return fig, ax
