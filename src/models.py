"""
models.py
=========

PyMC model definitions for Bayesian tensile strength prediction.

Two model classes are provided:

BayesianStrengthModel
    Single-phase Bayesian linear regression with weakly informative priors.
    Supports both NUTS (full MCMC) and ADVI (variational inference) inference
    backends for the tractability comparison study.

HierarchicalStrengthModel
    Multi-phase hierarchical (multilevel) model with partial pooling across
    alloy families. Intercepts are drawn from a common hyperprior, enabling
    shrinkage of noisy group-level estimates toward the population mean.

Both classes follow the same interface:
    model.build()   → construct the PyMC model graph
    model.sample()  → run inference and return an ArviZ InferenceData object
    model.predict() → posterior predictive samples on new covariates
    model.summary() → print ArviZ summary table
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


# ──────────────────────────────────────────────────────────────────────────────
# Model 1 — Bayesian Linear Regression (single phase)
# ──────────────────────────────────────────────────────────────────────────────


class BayesianStrengthModel:
    """Bayesian linear regression for single-phase steel tensile strength.

    The statistical model is:

        alpha   ~ Normal(500, 100)
        beta_T  ~ Normal(-0.5, 0.3)
        beta_C  ~ Normal(250, 100)
        sigma   ~ HalfNormal(50)

        mu_i    = alpha + beta_T * T_i + beta_C * C_i
        y_i     ~ Normal(mu_i, sigma)

    Prior justification
    -------------------
    - **alpha** ~ Normal(500, 100): A constructional steel at 0 °C and 0 wt%C
      (extrapolated) has a yield/tensile strength in the 400–600 MPa range.
      SD of 100 MPa is weakly informative, allowing the data to dominate.
    - **beta_T** ~ Normal(-0.5, 0.3): Thermally activated softening for steels
      is documented in the range -0.3 to -0.8 MPa/°C (Leslie, 1981). The prior
      is centred on the literature mean; SD of 0.3 spans the plausible range.
    - **beta_C** ~ Normal(250, 100): Carbon strengthening via solid-solution
      and pearlite formation. Literature values range 150–350 MPa/wt%C
      (Bhadeshia & Honeycombe, 2017, Chapter 2). SD 100 is weakly informative.
    - **sigma** ~ HalfNormal(50): ISO 6892-1 repeatability studies report
      tensile-testing scatter of 10–40 MPa; HalfNormal(50) assigns most
      probability below 80 MPa while remaining proper and non-negative.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns ``temperature``, ``carbon_content``,
        ``tensile_strength``.
    name : str, optional
        Name tag for the PyMC model context. Default is ``"bayesian_lr"``.

    Attributes
    ----------
    model : pm.Model or None
        PyMC model object, set after calling ``build()``.
    idata : az.InferenceData or None
        ArviZ InferenceData with posterior (and optionally posterior_predictive
        and log_likelihood groups), set after calling ``sample()``.

    Examples
    --------
    >>> from src.data_generator import MaterialDataGenerator
    >>> gen = MaterialDataGenerator(random_seed=42)
    >>> ds = gen.generate_single_phase(n_samples=200)
    >>> model = BayesianStrengthModel(ds.data)
    >>> model.build()
    >>> idata = model.sample(draws=2000, tune=1000)
    >>> model.summary()
    """

    def __init__(self, data: pd.DataFrame, name: str = "bayesian_lr") -> None:
        self.data = data.copy()
        self.name = name
        self.model: Optional[pm.Model] = None
        self.idata: Optional[az.InferenceData] = None

        # Pre-extract arrays for efficiency inside PyMC context
        self._T = self.data["temperature"].values.astype(float)
        self._C = self.data["carbon_content"].values.astype(float)
        self._y = self.data["tensile_strength"].values.astype(float)

    def build(self) -> pm.Model:
        """Construct the PyMC model graph.

        Returns
        -------
        pm.Model
            The constructed model object (also stored in ``self.model``).
        """
        with pm.Model(name=self.name) as model:
            # ── Priors ───────────────────────────────────────────────────────
            # Intercept: tensile strength at T=0 °C, C=0 wt% (MPa)
            alpha = pm.Normal(
                "alpha", mu=500.0, sigma=100.0
            )
            # Temperature coefficient (MPa / °C)
            beta_T = pm.Normal(
                "beta_T", mu=-0.5, sigma=0.3
            )
            # Carbon content coefficient (MPa / wt%C)
            beta_C = pm.Normal(
                "beta_C", mu=250.0, sigma=100.0
            )
            # Measurement noise SD (MPa); consistent with ISO 6892-1
            sigma = pm.HalfNormal(
                "sigma", sigma=50.0
            )

            # ── Deterministic mean ───────────────────────────────────────────
            mu = pm.Deterministic(
                "mu", alpha + beta_T * self._T + beta_C * self._C
            )

            # ── Likelihood ───────────────────────────────────────────────────
            y_obs = pm.Normal(  # noqa: F841
                "y_obs", mu=mu, sigma=sigma, observed=self._y
            )

        self.model = model
        return model

    def sample(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int = 42,
        return_inferencedata: bool = True,
    ) -> az.InferenceData:
        """Run NUTS sampling and return an ArviZ InferenceData object.

        Parameters
        ----------
        draws : int, optional
            Number of posterior samples per chain. Default is 2000.
        tune : int, optional
            Number of tuning (warm-up) steps per chain. Default is 1000.
        chains : int, optional
            Number of parallel chains. Default is 4.
        target_accept : float, optional
            Target Metropolis acceptance rate for NUTS dual averaging step
            size adaptation. Default is 0.9 (recommended for correlated
            posteriors; higher values slow sampling but improve mixing).
        random_seed : int, optional
            Seed passed to the sampler for reproducibility.
        return_inferencedata : bool, optional
            Return an ArviZ InferenceData object (always True for this class).

        Returns
        -------
        az.InferenceData
            InferenceData with posterior, sample_stats, observed_data,
            posterior_predictive, and log_likelihood groups.

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called before ``sample()``.
        """
        if self.model is None:
            raise RuntimeError("Call build() before sample().")

        with self.model:
            self.idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                return_inferencedata=True,
                progressbar=True,
            )
            # Extend with posterior predictive and log-likelihood for LOO-CV
            pm.sample_posterior_predictive(
                self.idata, extend_inferencedata=True, random_seed=random_seed
            )

        return self.idata

    def sample_advi(
        self,
        n_iterations: int = 50_000,
        random_seed: int = 42,
    ) -> az.InferenceData:
        """Run mean-field ADVI for comparison with NUTS.

        Automatic Differentiation Variational Inference (ADVI) optimises a
        mean-field Gaussian variational family by maximising the ELBO via
        stochastic gradient ascent. The result is an approximate posterior
        that trades accuracy for speed.

        Parameters
        ----------
        n_iterations : int, optional
            Maximum ELBO optimisation steps. Default is 50 000.
        random_seed : int, optional
            Random seed for gradient noise reproducibility.

        Returns
        -------
        az.InferenceData
            InferenceData with posterior group sampled from the variational
            approximation (10 000 draws).

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called before ``sample_advi()``.
        """
        if self.model is None:
            raise RuntimeError("Call build() before sample_advi().")

        # Rebuild model for ADVI (PyMC model state is affected after NUTS sampling)
        with pm.Model(name=self.name) as model_advi:
            # ── Priors ───────────────────────────────────────────────────────
            # Intercept: tensile strength at T=0 °C, C=0 wt% (MPa)
            alpha = pm.Normal(
                "alpha", mu=500.0, sigma=100.0
            )
            # Temperature coefficient (MPa / °C)
            beta_T = pm.Normal(
                "beta_T", mu=-0.5, sigma=0.3
            )
            # Carbon content coefficient (MPa / wt%C)
            beta_C = pm.Normal(
                "beta_C", mu=250.0, sigma=100.0
            )
            # Measurement noise SD (MPa); consistent with ISO 6892-1
            sigma = pm.HalfNormal(
                "sigma", sigma=50.0
            )

            # ── Deterministic mean ───────────────────────────────────────────
            mu = pm.Deterministic(
                "mu", alpha + beta_T * self._T + beta_C * self._C
            )

            # ── Likelihood ───────────────────────────────────────────────────
            y_obs = pm.Normal(  # noqa: F841
                "y_obs", mu=mu, sigma=sigma, observed=self._y
            )

            approx = pm.fit(
                n=n_iterations,
                method="advi",
                random_seed=random_seed,
            )
            idata_advi = approx.sample(10_000)

        return idata_advi

    def predict(
        self,
        new_temperature: np.ndarray,
        new_carbon: np.ndarray,
        n_samples: int = 2000,
    ) -> np.ndarray:
        """Draw posterior predictive samples at new covariate values.

        Parameters
        ----------
        new_temperature : np.ndarray, shape (n,)
            New temperature values in °C.
        new_carbon : np.ndarray, shape (n,)
            New carbon content values in wt%.
        n_samples : int, optional
            Number of posterior samples to use. Default is 2000.

        Returns
        -------
        np.ndarray, shape (n_samples, n)
            Posterior predictive samples of tensile strength (MPa).

        Raises
        ------
        RuntimeError
            If ``sample()`` has not been called before ``predict()``.
        """
        if self.idata is None:
            raise RuntimeError("Call sample() before predict().")

        posterior = self.idata.posterior
        # Flatten chains × draws
        # Note: variables are namespaced with model name (e.g., "bayesian_lr::alpha")
        alpha_samples = posterior[f"{self.name}::alpha"].values.reshape(-1)[:n_samples]
        beta_T_samples = posterior[f"{self.name}::beta_T"].values.reshape(-1)[:n_samples]
        beta_C_samples = posterior[f"{self.name}::beta_C"].values.reshape(-1)[:n_samples]
        sigma_samples = posterior[f"{self.name}::sigma"].values.reshape(-1)[:n_samples]

        rng = np.random.default_rng(0)
        mu_pred = (
            alpha_samples[:, None]
            + beta_T_samples[:, None] * new_temperature[None, :]
            + beta_C_samples[:, None] * new_carbon[None, :]
        )
        noise = rng.normal(0, sigma_samples[:, None], size=mu_pred.shape)
        return mu_pred + noise

    def summary(self, var_names: Optional[list[str]] = None) -> pd.DataFrame:
        """Print and return the ArviZ posterior summary.

        Parameters
        ----------
        var_names : list[str] or None, optional
            Variables to include. Default is ``["alpha", "beta_T",
            "beta_C", "sigma"]``.

        Returns
        -------
        pd.DataFrame
            ArviZ summary DataFrame with mean, SD, HDI, ESS, and R-hat.

        Raises
        ------
        RuntimeError
            If ``sample()`` has not been called.
        """
        if self.idata is None:
            raise RuntimeError("Call sample() before summary().")

        if var_names is None:
            var_names = ["alpha", "beta_T", "beta_C", "sigma"]

        # Prepend model name prefix used by PyMC when name is set
        prefixed = [f"{self.name}::{v}" for v in var_names]
        # Try with prefix, fall back to bare names
        try:
            summary_df = az.summary(self.idata, var_names=prefixed, round_to=4)
        except KeyError:
            summary_df = az.summary(self.idata, var_names=var_names, round_to=4)

        print(summary_df.to_string())
        return summary_df


# ──────────────────────────────────────────────────────────────────────────────
# Model 2 — Hierarchical Bayesian Model (multi-phase)
# ──────────────────────────────────────────────────────────────────────────────


class HierarchicalStrengthModel:
    """Hierarchical Bayesian model with partial pooling across alloy families.

    The model uses hyperpriors over alloy-family intercepts to implement
    partial pooling (Gelman & Hill, 2007). The group-level intercepts are
    drawn from a common Normal hyperprior, so groups with few observations
    are shrunk toward the global mean, reducing variance at the cost of a
    small bias — a favourable bias-variance tradeoff in practice.

    Mathematical specification
    --------------------------

    Hyperpriors (population level):
        mu_alpha    ~ Normal(500, 100)          [global mean intercept]
        sigma_alpha ~ HalfNormal(100)           [between-group SD]

    Group-level priors (partial pooling):
        alpha_g     ~ Normal(mu_alpha, sigma_alpha)   g ∈ {0, 1, 2}

    Shared (common-slope) priors:
        beta_T      ~ Normal(-0.5, 0.3)
        beta_C      ~ Normal(250, 100)
        sigma       ~ HalfNormal(50)

    Likelihood:
        mu_i = alpha_{g[i]} + beta_T * T_i + beta_C * C_i
        y_i  ~ Normal(mu_i, sigma)

    The shared-slope assumption (common beta_T, beta_C across groups) is
    scientifically justified by the universal thermally activated dislocation
    mechanism; the differing intercepts reflect differences in microstructural
    hardening density between phases.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns ``temperature``, ``carbon_content``,
        ``tensile_strength``, ``group_idx`` (integer 0, 1, 2).
    group_labels : list[str] or None, optional
        Human-readable labels for groups in display order. If None, defaults
        to ``["austenitic", "ferritic", "martensitic"]``.
    name : str, optional
        Name tag for the PyMC model context. Default is ``"hierarchical"``.

    Attributes
    ----------
    model : pm.Model or None
    idata : az.InferenceData or None

    Examples
    --------
    >>> from src.data_generator import MaterialDataGenerator
    >>> gen = MaterialDataGenerator(random_seed=42)
    >>> ds = gen.generate_multiphase()
    >>> hm = HierarchicalStrengthModel(ds.data)
    >>> hm.build()
    >>> idata = hm.sample(draws=2000, tune=1000)
    >>> hm.summary()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        group_labels: Optional[list[str]] = None,
        name: str = "hierarchical",
    ) -> None:
        self.data = data.copy()
        self.name = name
        self.group_labels = group_labels or ["austenitic", "ferritic", "martensitic"]
        self.n_groups = len(self.group_labels)
        self.model: Optional[pm.Model] = None
        self.idata: Optional[az.InferenceData] = None

        self._T = self.data["temperature"].values.astype(float)
        self._C = self.data["carbon_content"].values.astype(float)
        self._y = self.data["tensile_strength"].values.astype(float)
        self._g = self.data["group_idx"].values.astype(int)

    def build(self) -> pm.Model:
        """Construct the hierarchical PyMC model graph.

        Returns
        -------
        pm.Model
            Constructed model (also stored in ``self.model``).
        """
        coords = {"alloy": self.group_labels}

        with pm.Model(name=self.name, coords=coords) as model:
            # ── Data containers ──────────────────────────────────────────────
            group_idx = pm.Data("group_idx", self._g, dims="obs_id")

            # ── Hyperpriors (population level) ───────────────────────────────
            # Global mean intercept across alloy families (MPa)
            mu_alpha = pm.Normal(
                "mu_alpha", mu=500.0, sigma=100.0
            )
            # Between-group SD of intercepts (MPa)
            sigma_alpha = pm.HalfNormal(
                "sigma_alpha", sigma=100.0
            )

            # ── Group-level intercepts (partial pooling) ─────────────────────
            alpha_offset = pm.Normal(
                "alpha_offset", mu=0.0, sigma=1.0, dims="alloy"
            )
            # Group-specific intercepts (non-centred parametrisation)
            alpha = pm.Deterministic(
                "alpha",
                mu_alpha + sigma_alpha * alpha_offset,
                dims="alloy"
            )

            # ── Shared (common) slope priors ─────────────────────────────────
            # Temperature coefficient, shared across phases (MPa/°C)
            beta_T = pm.Normal(
                "beta_T", mu=-0.5, sigma=0.3
            )
            # Carbon content coefficient, shared (MPa/wt%C)
            beta_C = pm.Normal(
                "beta_C", mu=250.0, sigma=100.0
            )
            # Within-group residual noise (MPa)
            sigma = pm.HalfNormal(
                "sigma", sigma=50.0
            )

            # ── Deterministic mean ───────────────────────────────────────────
            mu = pm.Deterministic(
                "mu",
                alpha[group_idx] + beta_T * self._T + beta_C * self._C,
            )

            # ── Likelihood ───────────────────────────────────────────────────
            y_obs = pm.Normal(  # noqa: F841
                "y_obs", mu=mu, sigma=sigma, observed=self._y
            )

        self.model = model
        return model

    def sample(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int = 42,
    ) -> az.InferenceData:
        """Run NUTS sampling for the hierarchical model.

        Non-centred parametrisation (alpha = mu_alpha + sigma_alpha *
        alpha_offset) is used to avoid the funnel geometry that plagues
        centred hierarchical models when group sample sizes are small.
        See Betancourt & Girolami (2015) for theoretical background.

        Parameters
        ----------
        draws : int, optional
            Posterior draws per chain. Default is 2000.
        tune : int, optional
            Warm-up steps per chain. Default is 1000.
        chains : int, optional
            Number of parallel chains. Default is 4.
        target_accept : float, optional
            NUTS target acceptance rate. Default is 0.9.
        random_seed : int, optional
            Sampler seed. Default is 42.

        Returns
        -------
        az.InferenceData
            InferenceData with posterior, sample_stats, and
            posterior_predictive groups.

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called.
        """
        if self.model is None:
            raise RuntimeError("Call build() before sample().")

        with self.model:
            self.idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                return_inferencedata=True,
                progressbar=True,
            )
            pm.sample_posterior_predictive(
                self.idata, extend_inferencedata=True, random_seed=random_seed
            )

        return self.idata

    def predict(
        self,
        new_temperature: np.ndarray,
        new_carbon: np.ndarray,
        group_idx: int,
        n_samples: int = 2000,
    ) -> np.ndarray:
        """Draw posterior predictive samples for a specific alloy group.

        Parameters
        ----------
        new_temperature : np.ndarray, shape (n,)
            New temperature values in °C.
        new_carbon : np.ndarray, shape (n,)
            New carbon content values in wt%.
        group_idx : int
            Alloy group index (0=austenitic, 1=ferritic, 2=martensitic).
        n_samples : int, optional
            Number of posterior draws to use. Default is 2000.

        Returns
        -------
        np.ndarray, shape (n_samples, n)
            Posterior predictive tensile strength samples (MPa).

        Raises
        ------
        RuntimeError
            If ``sample()`` has not been called.
        """
        if self.idata is None:
            raise RuntimeError("Call sample() before predict().")

        posterior = self.idata.posterior
        # Note: variables are namespaced with model name (e.g., "hierarchical::alpha")
        alpha_s = posterior[f"{self.name}::alpha"].values.reshape(-1, self.n_groups)[:n_samples, group_idx]
        beta_T_s = posterior[f"{self.name}::beta_T"].values.reshape(-1)[:n_samples]
        beta_C_s = posterior[f"{self.name}::beta_C"].values.reshape(-1)[:n_samples]
        sigma_s = posterior[f"{self.name}::sigma"].values.reshape(-1)[:n_samples]

        rng = np.random.default_rng(0)
        mu_pred = (
            alpha_s[:, None]
            + beta_T_s[:, None] * new_temperature[None, :]
            + beta_C_s[:, None] * new_carbon[None, :]
        )
        noise = rng.normal(0, sigma_s[:, None], size=mu_pred.shape)
        return mu_pred + noise

    def summary(self, var_names: Optional[list[str]] = None) -> pd.DataFrame:
        """Print and return the posterior summary table.

        Parameters
        ----------
        var_names : list[str] or None, optional
            Variables to display. Defaults to the main model parameters.

        Returns
        -------
        pd.DataFrame
            ArviZ summary with mean, SD, HDI, ESS, R-hat.

        Raises
        ------
        RuntimeError
            If ``sample()`` has not been called.
        """
        if self.idata is None:
            raise RuntimeError("Call sample() before summary().")

        if var_names is None:
            var_names = ["mu_alpha", "sigma_alpha", "alpha", "beta_T", "beta_C", "sigma"]

        try:
            prefixed = [f"{self.name}::{v}" for v in var_names]
            summary_df = az.summary(self.idata, var_names=prefixed, round_to=4)
        except KeyError:
            summary_df = az.summary(self.idata, var_names=var_names, round_to=4)

        print(summary_df.to_string())
        return summary_df
