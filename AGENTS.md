# AGENTS.md

This file provides guidance to Qoder (qoder.com) when working with code in this repository.

## Project Overview

This is a Python implementation of the SAEM (Stochastic Approximation Expectation Maximization) algorithm for nonlinear mixed effects models, ported from the R saemix package. The library is used for parameter estimation in pharmacokinetic/pharmacodynamic (PK/PD) modeling and other applications involving hierarchical nonlinear models.

## Commands

### Testing
```bash
pytest                           # Run all tests
pytest tests/test_data.py       # Run specific test file
pytest -v --tb=short            # Verbose output with short traceback
pytest --cov=saemix             # Run with coverage
```

### Code Quality
```bash
black saemix/                   # Format code
isort saemix/                   # Sort imports
mypy saemix/                    # Type checking
```

### Build and Distribution
```bash
python -m build                 # Build package (creates dist/)
pip install -e .                # Install in editable/development mode
twine check dist/*              # Validate distribution files
```

### Running Examples
```bash
python examples/basic_example.py              # Basic usage example
python demo_estimation.py                      # Estimation demo
```

## Architecture

### Core Algorithm Flow

The SAEM algorithm execution follows this sequence:
1. **Initialization** (`saemix/algorithm/initialization.py`): Set up data structures, initial parameter values, and random effects
2. **Main Loop** (`saemix/algorithm/saem.py`): Iterative E-step and M-step updates
   - **E-step** (`saemix/algorithm/estep.py`): Simulate individual parameters using MCMC
   - **M-step** (`saemix/algorithm/mstep.py`): Update population parameters based on sufficient statistics
3. **Post-Processing**:
   - **MAP Estimation** (`saemix/algorithm/map_estimation.py`): Compute maximum a posteriori individual parameters
   - **FIM** (`saemix/algorithm/fim.py`): Calculate Fisher Information Matrix for standard errors
   - **Likelihood** (`saemix/algorithm/likelihood.py`): Compute log-likelihood using importance sampling or Gaussian quadrature

### Key Classes

- **SaemixData** (`saemix/data.py`): Data container managing longitudinal data with subject IDs, predictors, response, and optional covariates
- **SaemixModel** (`saemix/model.py`): Model specification including:
  - Structural model function `f(psi, id, xidep)` 
  - Parameter transformations (log-normal, probit, logit)
  - Error models (constant, proportional, combined, exponential)
  - Covariance structure for random effects
- **SaemixObject** (`saemix/results.py`): Result container with fitted parameters, predictions, residuals, and diagnostic information
- **saemix_control** (`saemix/control.py`): Algorithm control parameters (iterations, chains, convergence criteria, random seed)

### Critical Implementation Details

#### 0-based vs 1-based Indexing
The most important difference from R saemix is Python's 0-based indexing:
- Model functions access `xidep[:, 0]` for first predictor (R uses `xidep[,1]`)
- Parameter indexing: `psi[id, 0]` for first parameter (R uses `psi[id,1]`)
- Subject IDs are converted internally: user-facing IDs start at 1, but internal arrays use 0-based indexing

#### Random Number Generation
- The package uses `numpy.random.Generator` for reproducibility
- RNG is initialized from `seed` in control options and passed through all algorithm components
- Set `seed` in `saemix_control()` for reproducible results

#### Error Model Handling
- Exponential error models trigger automatic log-transformation of response data
- Original data is stored in `data.yorig` before transformation
- Multiple error models can be specified for different response types using the `ytype` column

#### Parameter Transformations
- Transformations are specified via `transform_par` in SaemixModel:
  - `0`: normal (no transformation)
  - `1`: log-normal (log transformation)
  - `2`: probit
  - `3`: logit
- Internal algorithms work in transformed space; `transphi()` utility converts between spaces

#### SAEM Mathematical Correctness (Fixed Issues)
Two critical mathematical issues were identified and fixed:

1. **Step Size Sequence** (`saemix/algorithm/initialization.py:174`):
   - **Fixed**: Changed from geometric decay `γₖ = γₖ₋₁ * α` to `γₖ = 1/k` after burn-in
   - **Theory**: SAEM requires ∑γₖ → ∞ and ∑γₖ² < ∞ for convergence
   - **Impact**: Ensures theoretical convergence guarantees are satisfied

2. **Omega Update** (`saemix/algorithm/mstep.py:296-311`):
   - **Fixed**: Changed to use accumulated sufficient statistics `Ω = suffStat["statphi2"]`
   - **Previous bug**: Was computing from current iteration only: `Ω = (ηᵀη)/n`
   - **Theory**: SAEM M-step must use stochastically approximated sufficient statistics, not raw samples
   - **Impact**: Properly implements the stochastic approximation; improves variance estimation stability

### Module Organization

```
saemix/
├── algorithm/          # Core SAEM algorithm components
│   ├── saem.py        # Main SAEM loop (run_saem)
│   ├── estep.py       # E-step: MCMC simulation of random effects
│   ├── mstep.py       # M-step: parameter updates
│   ├── initialization.py  # Algorithm initialization
│   ├── map_estimation.py  # MAP individual parameter estimation
│   ├── fim.py         # Fisher Information Matrix computation
│   ├── likelihood.py  # Log-likelihood calculation (IS, GQ methods)
│   ├── predict.py     # Predictions and residuals
│   └── conddist.py    # Conditional distribution estimation
├── data.py            # SaemixData class
├── model.py           # SaemixModel class
├── control.py         # Algorithm control options
├── results.py         # SaemixObject and SaemixRes result classes
├── main.py            # Main saemix() entry point
├── diagnostics.py     # Diagnostic plots (GOF, VPC, NPDE, residuals)
├── compare.py         # Model comparison (AIC, BIC)
├── simulation.py      # Simulation from fitted models
├── export.py          # Result export to CSV and files
└── utils.py           # Utility functions (transphi, cutoff, etc.)
```

### Testing Strategy

- **Property-based tests**: Use `hypothesis` for testing mathematical properties (e.g., `test_*_properties.py`)
- **Integration tests**: Full workflow tests with realistic models (`test_integration*.py`)
- **Regression tests**: Verify consistency with R saemix results (`test_regression.py`)
- **Build verification**: Ensure package builds correctly for PyPI (`test_build_verification.py`)

### Common Model Definition Pattern

```python
def structural_model(psi, id, xidep):
    """
    Model function signature required by saemix.
    
    Parameters:
    - psi: (n_subjects, n_params) individual parameters
    - id: (n_obs,) subject indices (0-based internally)
    - xidep: (n_obs, n_predictors) predictor variables
    
    Returns:
    - ypred: (n_obs,) predicted response
    """
    # Extract predictors (0-based indexing)
    x1 = xidep[:, 0]
    x2 = xidep[:, 1]
    
    # Extract individual parameters (0-based indexing)
    theta1 = psi[id, 0]
    theta2 = psi[id, 1]
    
    # Compute predictions
    ypred = ... 
    
    return ypred
```

### Dependencies

- **Required**: numpy, pandas, scipy
- **Optional**: matplotlib (for plotting functions)
- Plotting functions check for matplotlib availability via `_require_matplotlib()` helper

### Design Documentation

See `DESIGN.md` for comprehensive design rationale, including detailed R-to-Python migration considerations, data structure mappings, and API design decisions.