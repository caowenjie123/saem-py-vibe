"""
Pytest Configuration and Shared Fixtures for SAEMIX Tests

This module provides:
- Hypothesis profile configuration (ci/dev/debug)
- Shared test fixtures
- Test data generation strategies
"""

import os
import numpy as np
import pandas as pd
import pytest
from hypothesis import settings, Verbosity, strategies as st, Phase


# =============================================================================
# Hypothesis Profile Configuration
# =============================================================================

# CI profile: thorough testing with 100 examples
settings.register_profile(
    "ci",
    max_examples=100,
    deadline=None,
    suppress_health_check=[],
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target, Phase.shrink],
)

# Dev profile: faster iteration with 20 examples
settings.register_profile(
    "dev",
    max_examples=20,
    deadline=None,
    suppress_health_check=[],
)

# Debug profile: minimal examples with verbose output
settings.register_profile(
    "debug",
    max_examples=10,
    deadline=None,
    verbosity=Verbosity.verbose,
    suppress_health_check=[],
)

# Load profile based on environment variable, default to "dev"
_profile = os.getenv("HYPOTHESIS_PROFILE", "dev")
settings.load_profile(_profile)


# =============================================================================
# Test Data Generation Strategies
# =============================================================================


@st.composite
def valid_saemix_dataframe(
    draw,
    min_subjects=2,
    max_subjects=20,
    min_obs_per_subject=3,
    max_obs_per_subject=10,
    include_mdv=None,
    include_cens=None,
    include_occ=None,
    include_ytype=None,
):
    """
    Generate a valid SaemixData input DataFrame.

    Parameters
    ----------
    draw : hypothesis draw function
    min_subjects : int
        Minimum number of subjects
    max_subjects : int
        Maximum number of subjects
    min_obs_per_subject : int
        Minimum observations per subject
    max_obs_per_subject : int
        Maximum observations per subject
    include_mdv : bool or None
        If True, include MDV column. If None, randomly decide.
    include_cens : bool or None
        If True, include CENS column. If None, randomly decide.
    include_occ : bool or None
        If True, include OCC column. If None, randomly decide.
    include_ytype : bool or None
        If True, include YTYPE column. If None, randomly decide.

    Returns
    -------
    pd.DataFrame
        Valid DataFrame for SaemixData
    """
    n_subjects = draw(st.integers(min_value=min_subjects, max_value=max_subjects))
    n_obs_per_subject = draw(
        st.integers(min_value=min_obs_per_subject, max_value=max_obs_per_subject)
    )

    total_obs = n_subjects * n_obs_per_subject

    # Generate IDs (1-based)
    ids = np.repeat(np.arange(1, n_subjects + 1), n_obs_per_subject)

    # Generate times (sorted within each subject)
    times = np.tile(np.linspace(0, 24, n_obs_per_subject), n_subjects)

    # Generate DV values (positive, finite)
    dv = draw(
        st.lists(
            st.floats(
                min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False
            ),
            min_size=total_obs,
            max_size=total_obs,
        )
    )

    df = pd.DataFrame({"ID": ids, "TIME": times, "DV": dv})

    # Optionally add auxiliary columns
    if include_mdv is None:
        include_mdv = draw(st.booleans())
    if include_mdv:
        mdv_values = draw(
            st.lists(
                st.integers(min_value=0, max_value=1),
                min_size=total_obs,
                max_size=total_obs,
            )
        )
        df["MDV"] = mdv_values

    if include_cens is None:
        include_cens = draw(st.booleans())
    if include_cens:
        cens_values = draw(
            st.lists(
                st.integers(min_value=0, max_value=1),
                min_size=total_obs,
                max_size=total_obs,
            )
        )
        df["CENS"] = cens_values

    if include_occ is None:
        include_occ = draw(st.booleans())
    if include_occ:
        occ_values = draw(
            st.lists(
                st.integers(min_value=1, max_value=3),
                min_size=total_obs,
                max_size=total_obs,
            )
        )
        df["OCC"] = occ_values

    if include_ytype is None:
        include_ytype = draw(st.booleans())
    if include_ytype:
        ytype_values = draw(
            st.lists(
                st.integers(min_value=1, max_value=2),
                min_size=total_obs,
                max_size=total_obs,
            )
        )
        df["YTYPE"] = ytype_values

    return df


@st.composite
def transformation_inputs(draw, transform_type="any"):
    """
    Generate parameter transformation test inputs.

    Parameters
    ----------
    draw : hypothesis draw function
    transform_type : str
        One of 'log', 'logit', 'probit', or 'any'

    Returns
    -------
    np.ndarray
        Array of values suitable for the specified transformation
    """
    n = draw(st.integers(min_value=1, max_value=100))

    if transform_type == "any":
        transform_type = draw(st.sampled_from(["log", "logit", "probit", "none"]))

    if transform_type == "log":
        # Log transformation: input must be positive
        values = draw(
            st.lists(
                st.floats(
                    min_value=1e-10,
                    max_value=1e10,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=n,
                max_size=n,
            )
        )
    elif transform_type == "logit":
        # Logit transformation: input must be in (0, 1)
        values = draw(
            st.lists(
                st.floats(
                    min_value=1e-10,
                    max_value=1 - 1e-10,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=n,
                max_size=n,
            )
        )
    elif transform_type == "probit":
        # Probit transformation: input must be in (0, 1)
        values = draw(
            st.lists(
                st.floats(
                    min_value=1e-10,
                    max_value=1 - 1e-10,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=n,
                max_size=n,
            )
        )
    else:
        # No transformation: any finite values
        values = draw(
            st.lists(
                st.floats(
                    min_value=-10, max_value=10, allow_nan=False, allow_infinity=False
                ),
                min_size=n,
                max_size=n,
            )
        )

    return np.array(values)


@st.composite
def covariate_names(draw, min_covariates=1, max_covariates=5, include_missing=True):
    """
    Generate a list of covariate names, optionally including some that don't exist.

    Parameters
    ----------
    draw : hypothesis draw function
    min_covariates : int
        Minimum number of covariates
    max_covariates : int
        Maximum number of covariates
    include_missing : bool
        Whether to include some non-existent covariate names

    Returns
    -------
    tuple
        (list of covariate names, set of valid names, set of invalid names)
    """
    valid_names = ["AGE", "WT", "SEX", "RACE", "HT", "BMI", "CRCL", "ALB"]
    invalid_names = ["MISSING1", "MISSING2", "NOTFOUND", "INVALID"]

    n_valid = draw(
        st.integers(
            min_value=min_covariates, max_value=min(max_covariates, len(valid_names))
        )
    )
    selected_valid = draw(
        st.lists(
            st.sampled_from(valid_names),
            min_size=n_valid,
            max_size=n_valid,
            unique=True,
        )
    )

    if include_missing:
        n_invalid = draw(st.integers(min_value=0, max_value=2))
        selected_invalid = draw(
            st.lists(
                st.sampled_from(invalid_names),
                min_size=n_invalid,
                max_size=n_invalid,
                unique=True,
            )
        )
    else:
        selected_invalid = []

    # Combine and shuffle
    all_names = selected_valid + selected_invalid
    shuffled = draw(st.permutations(all_names))

    return list(shuffled), set(selected_valid), set(selected_invalid)


# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def simple_dataframe():
    """Create a simple test DataFrame."""
    return pd.DataFrame(
        {
            "ID": [1, 1, 1, 2, 2, 2],
            "TIME": [0, 1, 2, 0, 1, 2],
            "DV": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
        }
    )


@pytest.fixture
def dataframe_with_covariates():
    """Create a test DataFrame with covariates."""
    return pd.DataFrame(
        {
            "ID": [1, 1, 1, 2, 2, 2],
            "TIME": [0, 1, 2, 0, 1, 2],
            "DV": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
            "AGE": [30, 30, 30, 45, 45, 45],
            "WT": [70, 70, 70, 80, 80, 80],
        }
    )


@pytest.fixture
def dataframe_with_auxiliary():
    """Create a test DataFrame with auxiliary columns."""
    return pd.DataFrame(
        {
            "ID": [1, 1, 1, 2, 2, 2],
            "TIME": [0, 1, 2, 0, 1, 2],
            "DV": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
            "MDV": [0, 0, 0, 0, 0, 0],
            "CENS": [0, 0, 0, 0, 0, 0],
            "OCC": [1, 1, 1, 1, 1, 1],
            "YTYPE": [1, 1, 1, 1, 1, 1],
        }
    )


@pytest.fixture
def rng_seed():
    """Provide a fixed seed for reproducible tests."""
    return 12345


@pytest.fixture
def numpy_rng(rng_seed):
    """Create a numpy random generator with fixed seed."""
    return np.random.default_rng(rng_seed)
