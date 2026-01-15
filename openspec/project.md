# Project Context

## Purpose
Python implementation of the SAEM (Stochastic Approximation EM) algorithm for nonlinear mixed effects models, ported from the R saemix package. Focus on PK/PD parameter estimation, simulation, diagnostics, and model comparison.

## Tech Stack
- Python 3.8+
- numpy, pandas, scipy
- matplotlib (optional) for plotting
- packaging: setuptools, pyproject.toml
- testing: pytest, hypothesis
- code quality: black, isort, mypy

## Project Conventions

### Code Style
- snake_case for functions/variables; PascalCase for classes; UPPER_SNAKE for constants
- import order: standard library, third-party, local modules; no wildcard imports
- Black formatting, line length 88; isort profile black
- type annotate signatures; use Optional[...] for optional arrays
- NumPy-style docstrings with Parameters, Returns, Raises
- error handling: TypeError for wrong types, ValueError for invalid values
- RNG: use numpy.random.Generator and pass through call stack; seed via saemix_control
- prefer vectorized numpy operations over Python loops

### Architecture Patterns
- core package in `saemix/` with data, model, control, results, main, diagnostics, utils
- algorithm implementation in `saemix/algorithm/` (saem, estep, mstep, initialization, likelihood, fim, map_estimation, predict, conddist)
- typical flow: SaemixData + SaemixModel + saemix_control -> saemix() -> SaemixObject
- reference R implementation lives in `saemix-main/` for parity (not runtime code)

### Testing Strategy
- pytest is the primary runner; tests in `tests/`, files `test_*.py`
- fixtures live in `tests/conftest.py`
- Hypothesis property tests with profiles ci/dev/debug
- integration tests cover end-to-end estimation and diagnostics
- do not skip tests without justification

### Git Workflow
- not formally documented in repo; prefer small, focused commits and feature branches
- avoid rewriting shared history unless agreed

## Domain Context
- SAEM for nonlinear mixed effects (population PK/PD) models
- data is long format: one row per observation; requires subject ID, predictors, response; optional covariates
- model signature: `model(psi, id, xidep)` where psi=(n_subjects,n_params), id=(n_obs,) 0-based, xidep=(n_obs,n_predictors)
- parameter transforms via `transform_par`: 0 normal, 1 log, 2 probit, 3 logit
- error models: constant, proportional, combined, exponential

## Important Constraints
- all indexing is 0-based; subject IDs are remapped internally
- keep behavior aligned with R saemix where possible; avoid breaking API changes
- plotting requires matplotlib and should guard with `_require_matplotlib()`
- reproducibility: use numpy Generator and pass RNG through

## External Dependencies
- runtime: numpy>=1.20.0, pandas>=1.3.0, scipy>=1.7.0
- optional: matplotlib>=3.4.0 (plotting)
- dev: pytest, pytest-cov, hypothesis, black, isort, mypy, build, twine
- external services/APIs: none
