# AGENTS.md

This file provides guidance to agentic coding assistants working in this repository.

## Project Overview

Python implementation of the SAEM (Stochastic Approximation Expectation Maximization)
algorithm for nonlinear mixed effects models. This is a Python port of the R
`saemix` package used for PK/PD parameter estimation in hierarchical nonlinear models.

## Editor Rules

- No `.cursorrules`, `.cursor/rules/`, or `.github/copilot-instructions.md` found.
- Only follow rules defined in this file and other AGENTS.md files.

## Commands

### Installation
```bash
pip install -r requirements.txt
pip install -e .
pip install ".[dev]"
pip install ".[plot]"
```

### Testing
```bash
pytest
pytest tests/test_data.py
pytest tests/test_data.py::TestSaemixData::test_basic_creation
pytest -v --tb=short
pytest --cov=saemix
HYPOTHESIS_PROFILE=ci pytest tests/ -v --tb=short
HYPOTHESIS_PROFILE=dev pytest tests/
HYPOTHESIS_PROFILE=debug pytest tests/
```

Notes:
- Default pytest options are configured in `pyproject.toml` (`-v --tb=short`).
- Hypothesis profiles are defined in `tests/conftest.py` (`ci`, `dev`, `debug`).

### Code Quality
```bash
black saemix/
isort saemix/
mypy saemix/
```

### Build
```bash
python -m build
```

### Examples
```bash
python demo_estimation.py
python examples/basic_example.py
```

## Code Style Guidelines

### Imports
- Order: standard library → third-party → local modules.
- No wildcard imports.

```python
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from saemix.data import SaemixData
```

### Formatting
- Black formatter, line length 88 (`pyproject.toml`).
- isort profile `black` and line length 88.
- 4 spaces for indentation, no trailing whitespace.

### Type Annotations
- Use `typing` module: `Optional`, `List`, `Dict`, `Union`, `Callable`, `Tuple`.
- Annotate function signatures with parameter and return types.
- Use `Optional[np.ndarray]` or `None` defaults for optional arrays.

```python
def run_saem(
    Dargs: Dict[str, Any],
    Uargs: Dict[str, Any],
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    ...
```

### Naming Conventions
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

### Error Handling
- Validate inputs with `isinstance()` checks.
- Raise `TypeError` for wrong types.
- Raise `ValueError` for invalid values (bounds, shapes, incompatible inputs).
- Use descriptive error messages.

```python
if not isinstance(model, SaemixModel):
    raise TypeError("model must be a SaemixModel instance")
```

### Random Number Generation
- Use `numpy.random.Generator` for reproducibility.
- Pass RNG through function calls instead of creating new ones.
- Initialize from `seed` in `saemix_control()` or accept user-provided RNG.

```python
rng: Optional[np.random.Generator] = None
if rng is None:
    rng = np.random.default_rng(seed)
```

### Documentation
- Use NumPy-style docstrings with Parameters, Returns, Raises sections.
- Include type information in parameter descriptions.

```python
def transphi(phi, tr, verbose: bool = False):
    """
    Transform phi (untransformed) to psi (transformed) parameters.

    Parameters
    ----------
    phi : np.ndarray
        Untransformed parameter matrix
    tr : np.ndarray
        Transformation type vector (0=normal, 1=log, 2=probit, 3=logit)
    verbose : bool, optional
        Whether to output warnings (default False)

    Returns
    -------
    np.ndarray
        Transformed parameter matrix

    Raises
    ------
    ValueError
        If transformation produces overflow (Inf values)
    """
```

## Critical Implementation Details

### 0-based Indexing
Model functions use 0-based indexing (critical difference from R saemix):
- Predictors: `xidep[:, 0]` for first column.
- Parameters: `psi[id, 0]` for first parameter.
- Subject IDs: internal arrays use 0-based; user-facing IDs start at 1.

### Model Function Signature
```python
def structural_model(psi: np.ndarray, id: np.ndarray, xidep: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    psi : (n_subjects, n_params) individual parameters
    id : (n_obs,) subject indices (0-based)
    xidep : (n_obs, n_predictors) predictor variables

    Returns
    -------
    ypred : (n_obs,) predicted response
    """
    x1 = xidep[:, 0]
    theta1 = psi[id, 0]
    return theta1 * np.exp(-theta2 * x1)
```

### Parameter Transformations
Specified via `transform_par` in `SaemixModel`:
- `0`: normal (no transformation)
- `1`: log-normal (exp transformation)
- `2`: probit (norm.cdf transformation)
- `3`: logit
- Use `transphi()` to convert between spaces.

### Dependencies
- Required: numpy>=1.20.0, pandas>=1.3.0, scipy>=1.7.0
- Optional plotting: matplotlib>=3.4.0
- Check matplotlib availability with `_require_matplotlib()` helper.

## Architecture

- `saemix/algorithm/`: Core SAEM algorithm (`saem.py`, `estep.py`, `mstep.py`)
- `saemix/data.py`: `SaemixData` data container and validation
- `saemix/model.py`: `SaemixModel` model specification
- `saemix/control.py`: `saemix_control()` options
- `saemix/results.py`: `SaemixObject` result container
- `saemix/main.py`: Main `saemix()` entry point
- `saemix/diagnostics.py`: Diagnostics plots

See `DESIGN.md` for R-to-Python migration rationale and design details.
