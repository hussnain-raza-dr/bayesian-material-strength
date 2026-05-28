"""
data_generator.py
=================

Synthetic data generation for steel alloy tensile strength modeling.

The data-generating process is designed to reflect the physical metallurgy
of low-to-medium alloy steels in the temperature range 20–300 °C, following
relationships documented in Leslie (1981) and Bhadeshia & Honeycombe (2017).

The tensile strength model used for data generation is:

    sigma_y = alpha + beta_T * T + beta_C * C + epsilon

where:
    alpha   : alloy-family intercept (MPa)
    beta_T  : temperature coefficient (MPa / °C),  typically -0.4 to -0.7
    beta_C  : carbon coefficient    (MPa / wt%C),  typically +150 to +350
    epsilon : Gaussian noise ~ N(0, sigma^2), sigma ~ 20-40 MPa
              consistent with ISO 6892-1 round-robin repeatability data

The multi-phase generator adds a group-level random intercept drawn from
a common hyperprior, enabling hierarchical partial pooling in Model 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Data container
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class SteelDataset:
    """Container for generated steel alloy strength data.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns: ``temperature``, ``carbon_content``,
        ``tensile_strength``, and optionally ``alloy_phase``.
    true_params : dict
        Ground-truth parameters used in the data-generating process.
        Useful for posterior recovery checks.
    description : str
        Human-readable description of the dataset.
    """

    data: pd.DataFrame
    true_params: dict
    description: str = ""

    def __repr__(self) -> str:  # noqa: D105
        n = len(self.data)
        cols = list(self.data.columns)
        return f"SteelDataset(n={n}, columns={cols})"


# ──────────────────────────────────────────────────────────────────────────────
# Main generator class
# ──────────────────────────────────────────────────────────────────────────────


class MaterialDataGenerator:
    """Generate synthetic tensile strength data for steel alloy systems.

    This class provides three generation modes:

    1. **Single-phase** (``generate_single_phase``): one alloy family with
       Gaussian noise; suitable for Bayesian linear regression.
    2. **Multi-phase** (``generate_multiphase``): three alloy families
       (austenitic, ferritic, martensitic) with shared physics but distinct
       intercepts; suitable for hierarchical modeling.
    3. **Noise augmentation** (``add_measurement_noise``): post-hoc addition
       of correlated measurement noise to simulate sensor drift.

    Parameters
    ----------
    random_seed : int, optional
        Seed for the NumPy random number generator. Default is 42.

    Attributes
    ----------
    rng : np.random.Generator
        Seeded NumPy random generator (PCG64 algorithm).

    Examples
    --------
    >>> gen = MaterialDataGenerator(random_seed=42)
    >>> dataset = gen.generate_single_phase(n_samples=200)
    >>> dataset.data.head()

    Notes
    -----
    True parameter values are chosen to be consistent with published data
    for constructional steels (Leslie, 1981, Chapter 4; ASTM A36 / EN 10025).
    """

    # ── Physical plausibility bounds ────────────────────────────────────────
    _T_MIN: float = 20.0    # °C  — ambient
    _T_MAX: float = 300.0   # °C  — below creep regime for most steels
    _C_MIN: float = 0.05    # wt% — hypo-eutectoid lower bound
    _C_MAX: float = 0.60    # wt% — hypo-eutectoid upper bound

    # ── Alloy family parameters (ground truth) ───────────────────────────────
    _ALLOY_PARAMS: dict[str, dict] = {
        "austenitic": {
            "alpha": 490.0,   # MPa — intercept at T=0, C=0 (extrapolated)
            "beta_T": -0.45,  # MPa/°C
            "beta_C": 200.0,  # MPa/wt%C
            "sigma": 22.0,    # MPa — measurement noise
        },
        "ferritic": {
            "alpha": 380.0,
            "beta_T": -0.55,
            "beta_C": 260.0,
            "sigma": 28.0,
        },
        "martensitic": {
            "alpha": 780.0,
            "beta_T": -0.65,
            "beta_C": 310.0,
            "sigma": 35.0,
        },
    }

    def __init__(self, random_seed: int = 42) -> None:
        self.rng = np.random.default_rng(random_seed)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _sample_covariates(
        self,
        n_samples: int,
        t_range: tuple[float, float] = (_T_MIN, _T_MAX),
        c_range: tuple[float, float] = (_C_MIN, _C_MAX),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample temperature and carbon content uniformly over given ranges.

        Parameters
        ----------
        n_samples : int
            Number of observations to generate.
        t_range : tuple[float, float]
            (min, max) temperature in °C.
        c_range : tuple[float, float]
            (min, max) carbon content in wt%.

        Returns
        -------
        temperature : np.ndarray, shape (n_samples,)
        carbon : np.ndarray, shape (n_samples,)
        """
        temperature = self.rng.uniform(t_range[0], t_range[1], size=n_samples)
        carbon = self.rng.uniform(c_range[0], c_range[1], size=n_samples)
        return temperature, carbon

    def _compute_strength(
        self,
        temperature: np.ndarray,
        carbon: np.ndarray,
        alpha: float,
        beta_T: float,
        beta_C: float,
        sigma: float,
    ) -> np.ndarray:
        """Apply the linear strength model with additive Gaussian noise.

        Parameters
        ----------
        temperature : np.ndarray
            Temperature values in °C.
        carbon : np.ndarray
            Carbon content in wt%.
        alpha : float
            Intercept (MPa).
        beta_T : float
            Temperature slope (MPa / °C).
        beta_C : float
            Carbon slope (MPa / wt%C).
        sigma : float
            Noise standard deviation (MPa).

        Returns
        -------
        np.ndarray
            Tensile strength values (MPa).
        """
        mu = alpha + beta_T * temperature + beta_C * carbon
        noise = self.rng.normal(0.0, sigma, size=len(temperature))
        return mu + noise

    # ── Public API ───────────────────────────────────────────────────────────

    def generate_single_phase(
        self,
        n_samples: int = 200,
        alloy: str = "ferritic",
        t_range: tuple[float, float] = (20.0, 300.0),
        c_range: tuple[float, float] = (0.05, 0.60),
    ) -> SteelDataset:
        """Generate single-phase steel tensile strength data.

        Produces a dataset suitable for Bayesian linear regression (Model 1).
        Covariates (temperature, carbon content) are sampled independently
        from uniform distributions; strength follows a linear model with
        additive Gaussian noise (see module docstring for parameterisation).

        Parameters
        ----------
        n_samples : int, optional
            Number of synthetic observations. Default is 200.
        alloy : str, optional
            Alloy family to use. Must be one of ``"austenitic"``,
            ``"ferritic"``, or ``"martensitic"``. Default is ``"ferritic"``.
        t_range : tuple[float, float], optional
            Temperature range (min, max) in °C. Default is (20, 300).
        c_range : tuple[float, float], optional
            Carbon content range (min, max) in wt%. Default is (0.05, 0.60).

        Returns
        -------
        SteelDataset
            Container with DataFrame (columns: ``temperature``,
            ``carbon_content``, ``tensile_strength``) and ground-truth
            parameter dictionary.

        Raises
        ------
        ValueError
            If ``alloy`` is not a recognised alloy family name.

        Examples
        --------
        >>> gen = MaterialDataGenerator(random_seed=0)
        >>> ds = gen.generate_single_phase(n_samples=150, alloy="martensitic")
        >>> ds.data.describe()
        """
        if alloy not in self._ALLOY_PARAMS:
            raise ValueError(
                f"Unknown alloy '{alloy}'. "
                f"Choose from: {list(self._ALLOY_PARAMS.keys())}"
            )

        params = self._ALLOY_PARAMS[alloy]
        temperature, carbon = self._sample_covariates(n_samples, t_range, c_range)
        strength = self._compute_strength(temperature, carbon, **params)

        df = pd.DataFrame(
            {
                "temperature": temperature,
                "carbon_content": carbon,
                "tensile_strength": strength,
            }
        )

        return SteelDataset(
            data=df,
            true_params={**params, "alloy": alloy},
            description=(
                f"Single-phase {alloy} steel, n={n_samples}, "
                f"T∈[{t_range[0]}, {t_range[1]}] °C, "
                f"C∈[{c_range[0]}, {c_range[1]}] wt%"
            ),
        )

    def generate_multiphase(
        self,
        n_per_group: dict[str, int] | None = None,
        t_range: tuple[float, float] = (20.0, 300.0),
        c_range: tuple[float, float] = (0.05, 0.60),
    ) -> SteelDataset:
        """Generate multi-phase steel dataset for hierarchical modeling.

        Produces observations from three alloy families with shared covariate
        structure but distinct intercepts. The group-level intercepts are drawn
        from the fixed parameters in ``_ALLOY_PARAMS``; a hierarchical model
        will attempt to recover the common mean and between-group variance.

        Parameters
        ----------
        n_per_group : dict[str, int] or None, optional
            Number of observations per alloy family. Keys must be
            ``"austenitic"``, ``"ferritic"``, ``"martensitic"``.
            If None, defaults to ``{"austenitic": 80, "ferritic": 100,
            "martensitic": 60}`` — deliberately unequal to test partial
            pooling under data imbalance.
        t_range : tuple[float, float], optional
            Temperature range in °C. Default is (20, 300).
        c_range : tuple[float, float], optional
            Carbon content range in wt%. Default is (0.05, 0.60).

        Returns
        -------
        SteelDataset
            Container with DataFrame (columns: ``temperature``,
            ``carbon_content``, ``tensile_strength``, ``alloy_phase``,
            ``group_idx``) and nested ground-truth parameter dictionary.

        Examples
        --------
        >>> gen = MaterialDataGenerator(random_seed=42)
        >>> ds = gen.generate_multiphase(n_per_group={"austenitic": 50,
        ...                                            "ferritic": 50,
        ...                                            "martensitic": 50})
        >>> ds.data["alloy_phase"].value_counts()
        """
        if n_per_group is None:
            n_per_group = {"austenitic": 80, "ferritic": 100, "martensitic": 60}

        frames: list[pd.DataFrame] = []
        true_params: dict[str, dict] = {}

        for group_idx, (alloy, n) in enumerate(n_per_group.items()):
            if alloy not in self._ALLOY_PARAMS:
                raise ValueError(
                    f"Unknown alloy '{alloy}'. "
                    f"Choose from: {list(self._ALLOY_PARAMS.keys())}"
                )
            params = self._ALLOY_PARAMS[alloy]
            temperature, carbon = self._sample_covariates(n, t_range, c_range)
            strength = self._compute_strength(temperature, carbon, **params)

            frame = pd.DataFrame(
                {
                    "temperature": temperature,
                    "carbon_content": carbon,
                    "tensile_strength": strength,
                    "alloy_phase": alloy,
                    "group_idx": group_idx,
                }
            )
            frames.append(frame)
            true_params[alloy] = {**params, "group_idx": group_idx}

        df = pd.concat(frames, ignore_index=True)
        n_total = len(df)

        return SteelDataset(
            data=df,
            true_params=true_params,
            description=(
                f"Multi-phase steel dataset, n_total={n_total}, "
                f"groups={list(n_per_group.keys())}, "
                f"T∈[{t_range[0]}, {t_range[1]}] °C"
            ),
        )

    def add_measurement_noise(
        self,
        dataset: SteelDataset,
        additional_sigma: float = 10.0,
        drift_coefficient: float = 0.002,
    ) -> SteelDataset:
        """Augment an existing dataset with correlated measurement noise.

        Simulates sensor drift (linear in temperature) plus additional i.i.d.
        Gaussian noise, representing realistic degradation of a tensile-testing
        machine's load cell over a test campaign.

        Parameters
        ----------
        dataset : SteelDataset
            Input dataset to augment. The original ``tensile_strength`` column
            is overwritten with the noisy version; the true value is preserved
            in a new ``tensile_strength_clean`` column.
        additional_sigma : float, optional
            Standard deviation of additional i.i.d. noise in MPa. Default 10.
        drift_coefficient : float, optional
            Linear drift rate in MPa/°C. Default is 0.002.

        Returns
        -------
        SteelDataset
            New dataset with augmented noise. True parameters unchanged.

        Examples
        --------
        >>> gen = MaterialDataGenerator()
        >>> ds = gen.generate_single_phase()
        >>> ds_noisy = gen.add_measurement_noise(ds, additional_sigma=15.0)
        """
        df = dataset.data.copy()
        n = len(df)

        drift = drift_coefficient * df["temperature"].values
        extra_noise = self.rng.normal(0.0, additional_sigma, size=n)

        df["tensile_strength_clean"] = df["tensile_strength"].copy()
        df["tensile_strength"] = df["tensile_strength"] + drift + extra_noise

        return SteelDataset(
            data=df,
            true_params=dataset.true_params,
            description=dataset.description
            + f" [+drift={drift_coefficient} MPa/°C, +noise={additional_sigma} MPa]",
        )
