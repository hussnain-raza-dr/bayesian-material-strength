"""
bayesian-material-strength
==========================

Bayesian inference for tensile strength prediction in steel alloys.
Provides data generation, PyMC model definitions, and visualization
utilities for probabilistic material property modeling.

Modules
-------
data_generator
    Synthetic data generation for single-phase and multi-phase steel systems.
models
    BayesianStrengthModel and HierarchicalStrengthModel PyMC model classes.
visualization
    Reusable plotting functions for posterior, predictive, and diagnostic analysis.
"""

__version__ = "1.0.0"
__author__ = "Hussnain Raza"
__affiliation__ = "TU Bergakademie Freiberg"

from .data_generator import MaterialDataGenerator
from .models import BayesianStrengthModel, HierarchicalStrengthModel
from .visualization import (
    plot_posterior,
    plot_predictive_intervals,
    plot_group_comparison,
    plot_tractability_comparison,
)

__all__ = [
    "MaterialDataGenerator",
    "BayesianStrengthModel",
    "HierarchicalStrengthModel",
    "plot_posterior",
    "plot_predictive_intervals",
    "plot_group_comparison",
    "plot_tractability_comparison",
]
