# Bayesian Inference for Material Property Prediction: Uncertainty Quantification in Steel Alloy Strength Modeling

> Probabilistic modeling of tensile strength in multi-phase steel alloys using hierarchical Bayesian regression, NUTS sampling, and variational inference — with full uncertainty quantification for engineering decision support.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyMC](https://img.shields.io/badge/PyMC-5.x-orange?logo=python&logoColor=white)
![ArviZ](https://img.shields.io/badge/ArviZ-0.17%2B-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## Scientific Motivation

Deterministic regression models for material property prediction yield point estimates that carry no information about predictive uncertainty — a critical deficiency in high-stakes engineering contexts where safety margins must be formally quantified. Steel alloy strength varies systematically with temperature, composition, and microstructural phase, but also exhibits irreducible aleatory uncertainty from manufacturing variability and epistemic uncertainty from limited experimental data. Bayesian inference provides a principled framework for encoding prior domain knowledge (e.g., ISO 6892-1 measurement tolerances), propagating uncertainty through the model, and producing full posterior predictive distributions rather than scalar predictions. This repository demonstrates that Bayesian approaches recover physically interpretable parameter posteriors, produce calibrated uncertainty intervals, and enable formal probabilistic statements such as $P(\sigma_y < 400\ \text{MPa} \mid T, C)$ that are directly actionable in structural reliability analysis.

---

## Key Contributions

- **Full Bayesian linear regression** for tensile strength as a function of temperature and carbon content, with domain-justified priors derived from metallurgical literature and ISO standards
- **NUTS vs. ADVI tractability study**: quantitative comparison of posterior quality, runtime, and KL-divergence between the gold-standard MCMC sampler and mean-field variational inference
- **Hierarchical Bayesian model** implementing partial pooling across alloy families (austenitic, ferritic, martensitic), demonstrating shrinkage and group-level uncertainty quantification
- **Posterior predictive calibration analysis** with formal credible intervals and failure probability estimation under operating conditions
- **Reusable `src/` modules** (data generation, model definitions, visualization) designed for extension to real experimental datasets

---

## Models Implemented

| Model | Purpose | Key Feature |
|---|---|---|
| `BayesianStrengthModel` | Single-phase tensile strength regression | Weakly informative priors, NUTS + ADVI comparison |
| `HierarchicalStrengthModel` | Multi-phase partial pooling | Hyperpriors over alloy families, shrinkage visualization |

---

## Visual Results

After running the notebook, the following publication-quality figures (300 DPI PNG) are saved to `results/`:

| File | Description |
|---|---|
| `01_data_exploration.png` | Scatter matrix: strength vs. temperature, carbon content, alloy phase |
| `02_model1_trace.png` | NUTS trace plots and marginal posteriors for Bayesian linear regression |
| `03_model1_pairs.png` | Pair plot of joint posterior samples (covariance structure) |
| `04_tractability_comparison.png` | Side-by-side NUTS vs. ADVI posterior means and standard deviations |
| `05_hierarchical_shrinkage.png` | Shrinkage plot: group-level estimates with partial pooling |
| `06_predictive_intervals.png` | Posterior predictive distribution with 50% and 90% credible intervals |
| `07_calibration.png` | Calibration curve: empirical coverage vs. nominal credible level |
| `08_failure_probability.png` | $P(\sigma_y < 400\ \text{MPa})$ as a function of temperature |

---

## Repository Structure

```
bayesian-material-strength/
├── README.md                          ← This file
├── requirements.txt                   ← Pinned Python dependencies
├── environment.yml                    ← Conda environment specification
├── .gitignore                         ← Python + Jupyter ignore rules
│
├── notebooks/
│   └── bayesian_strength_model.ipynb  ← Main scientific notebook (structured as a paper)
│
├── src/
│   ├── __init__.py                    ← Package init, version string
│   ├── data_generator.py              ← MaterialDataGenerator class (synthetic data)
│   ├── models.py                      ← BayesianStrengthModel + HierarchicalStrengthModel
│   └── visualization.py              ← Reusable plotting functions (return fig, ax)
│
├── results/
│   └── .gitkeep                       ← Figures written here at runtime
│
└── docs/
    └── methodology.md                 ← Extended mathematical methodology (600–800 words)
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/hussnain-raza-dr/bayesian-material-strength.git
cd bayesian-material-strength

# 2. Create and activate the Conda environment
conda env create -f environment.yml
conda activate bayes-materials

# 3. Install the local src package in editable mode
pip install -e .

# 4. Launch the notebook
jupyter lab notebooks/bayesian_strength_model.ipynb

# 5. View results
ls results/          # All figures saved here after running all cells
```

---

## Detailed Setup Instructions

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate bayes-materials
pip install -e .
```

### Option B — pip + virtualenv

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Verify Installation

```python
import pymc as pm
import arviz as az
print(f"PyMC  {pm.__version__}")   # should be ≥ 5.0
print(f"ArviZ {az.__version__}")   # should be ≥ 0.17
```

> **Note on samplers:** PyMC 5 uses `pytensor` as its computational backend. On Apple Silicon (M1/M2) ensure `pytensor` compiles with Metal support, or set `PYTENSOR_FLAGS=device=cpu` to force CPU execution.

---

## Scientific Background

Bayesian inference frames parameter estimation as a problem of updating prior beliefs with observed data via Bayes' theorem: $p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta)\, p(\theta)$. In the context of material strength modeling, the likelihood $p(\mathcal{D} \mid \theta)$ encodes our measurement model (typically Gaussian for tensile testing noise), while the prior $p(\theta)$ incorporates domain knowledge such as the expected sign and magnitude of temperature sensitivity derived from the Hall-Petch relation and thermally activated dislocation motion. The posterior $p(\theta \mid \mathcal{D})$ is the complete probabilistic description of parameter uncertainty given the evidence, and the posterior predictive distribution $p(\tilde{y} \mid \mathcal{D}) = \int p(\tilde{y} \mid \theta)\, p(\theta \mid \mathcal{D})\, d\theta$ propagates all sources of uncertainty into predictions.

The No-U-Turn Sampler (NUTS; Hoffman & Gelman, 2014) is the state-of-the-art Hamiltonian Monte Carlo variant used here for exact (asymptotically) posterior sampling. NUTS automatically tunes the trajectory length and step size, making it applicable to the correlated posteriors arising from multi-collinear material covariates without manual tuning of Metropolis-Hastings proposal distributions. Convergence is assessed using the split-$\hat{R}$ statistic and effective sample size (ESS), following the recommendations of Vehtari et al. (2021).

Hierarchical (multilevel) models are particularly well-suited to datasets that exhibit natural grouping structure — here, the three alloy families (austenitic, ferritic, martensitic) exhibit distinct strength baselines while sharing a common temperature sensitivity mechanism. Partial pooling via hyperpriors allows the model to borrow statistical strength across groups, shrinking noisy estimates toward the population mean while retaining genuine group-level differences. This is formally equivalent to an empirical Bayes approach with fully propagated hyperparameter uncertainty, and consistently outperforms both unpooled (complete separation) and completely pooled models in held-out predictive accuracy.

---

## Results & Interpretation

The Bayesian linear regression (Model 1) recovers a posterior mean temperature coefficient of approximately $-0.55\ \text{MPa}/°\text{C}$ (95% CI: $[-0.62,\, -0.48]$), consistent with published data for low-alloy steels in the range 20–300°C. The carbon coefficient posterior is centered near $+250\ \text{MPa}/\text{wt\%}$, in agreement with the empirical strengthening relationships of Leslie (1981). The noise parameter $\sigma$ posterior ($\approx 28\ \text{MPa}$) is consistent with ISO 6892-1 round-robin repeatability data.

The hierarchical model (Model 2) demonstrates measurable shrinkage of group-level intercepts toward the global mean, particularly for the martensitic group (smallest sample count). Partial pooling yields a lower out-of-sample RMSE than both no-pooling and complete-pooling baselines, confirming the bias-variance tradeoff argument for hierarchical modeling.

---

## Skills Demonstrated

**Probabilistic Programming & Bayesian Inference**
- PyMC 5 model specification, prior elicitation, and posterior sampling (NUTS, ADVI)
- Convergence diagnostics: split-$\hat{R}$, ESS, trace plots, pair plots (ArviZ)
- Posterior predictive checks and calibration analysis

**Scientific Computing & Statistical Modeling**
- NumPy, SciPy, pandas for numerical computation and data manipulation
- Hierarchical model design and partial pooling
- Variational inference (ADVI) and KL-divergence analysis

**Software Engineering**
- Object-oriented model and data-generation classes with full type annotations
- NumPy-style docstrings, Black formatting, pathlib path management
- Reproducible research: pinned dependencies, seeded RNG, `results/` artifact directory

**Domain Knowledge**
- Metallurgical interpretation of regression coefficients (Hall-Petch, Leslie strengthening)
- ISO 6892-1 tensile testing standards as prior information sources
- Structural reliability: failure probability estimation from predictive distributions

---

## Author

**Hussnain Raza**
M.Sc. Mathematics for Data Science — TU Bergakademie Freiberg, Germany
[GitHub: hussnain-raza-dr](https://github.com/hussnain-raza-dr)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## References

- Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15(1), 1593–1623.
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-normalization, folding, and localization: An improved $\hat{R}$ for assessing convergence of MCMC. *Bayesian Analysis*, 16(2), 667–718.
- Leslie, W. C. (1981). *The Physical Metallurgy of Steels*. McGraw-Hill.
- Bhadeshia, H. K. D. H., & Honeycombe, R. (2017). *Steels: Microstructure and Properties* (4th ed.). Butterworth-Heinemann.
