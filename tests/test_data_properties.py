"""
Property-Based Tests for SaemixData Auxiliary Column Handling

This module tests the robustness of auxiliary column (mdv, cens, occ, ytype) handling
in the SaemixData class using property-based testing with Hypothesis.

Feature: saemix-robustness-optimization
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, assume

from saemix.data import SaemixData
from tests.conftest import valid_saemix_dataframe


class TestAuxiliaryColumnProperties:
    """Property-based tests for auxiliary column handling."""

    @given(
        df=valid_saemix_dataframe(
            min_subjects=2,
            max_subjects=10,
            min_obs_per_subject=3,
            max_obs_per_subject=5,
            include_mdv=True,
            include_cens=True,
            include_occ=True,
            include_ytype=True,
        )
    )
    @settings(max_examples=100)
    def test_auxiliary_column_mapping_preservation(self, df):
        """
        Feature: saemix-robustness-optimization
        Property 1: Auxiliary Column Mapping Preservation
        Validates: Requirements 1.1

        For any valid DataFrame with auxiliary columns (mdv, cens, occ, ytype)
        and corresponding column name specifications, when SaemixData processes
        the data, the auxiliary columns SHALL be correctly mapped and accessible
        after all internal transformations.
        """
        # Ensure all MDV values are 0 so no rows are filtered out
        df = df.copy()
        df["MDV"] = 0

        # Store original auxiliary values before processing
        df["MDV"].values.copy()
        df["CENS"].values.copy()
        df["OCC"].values.copy()
        df["YTYPE"].values.copy()

        # Create SaemixData with auxiliary column specifications
        saemix_data = SaemixData(
            name_data=df,
            name_group="ID",
            name_predictors=["TIME"],
            name_response="DV",
            name_mdv="MDV",
            name_cens="CENS",
            name_occ="OCC",
            name_ytype="YTYPE",
            verbose=False,
        )

        # Verify internal columns exist and are accessible
        assert "mdv" in saemix_data.data.columns, "Internal 'mdv' column should exist"
        assert "cens" in saemix_data.data.columns, "Internal 'cens' column should exist"
        assert "occ" in saemix_data.data.columns, "Internal 'occ' column should exist"
        assert (
            "ytype" in saemix_data.data.columns
        ), "Internal 'ytype' column should exist"

        # Verify the values are preserved (accounting for sorting)
        # Since data is sorted by ID and TIME, we need to compare sorted values
        sorted_df = df.sort_values(["ID", "TIME"]).reset_index(drop=True)

        # The processed data should have the same auxiliary values as the sorted original
        np.testing.assert_array_equal(
            saemix_data.data["mdv"].values,
            sorted_df["MDV"].values,
            err_msg="MDV values should be preserved after processing",
        )
        np.testing.assert_array_equal(
            saemix_data.data["cens"].values,
            sorted_df["CENS"].values,
            err_msg="CENS values should be preserved after processing",
        )
        np.testing.assert_array_equal(
            saemix_data.data["occ"].values,
            sorted_df["OCC"].values,
            err_msg="OCC values should be preserved after processing",
        )
        np.testing.assert_array_equal(
            saemix_data.data["ytype"].values,
            sorted_df["YTYPE"].values,
            err_msg="YTYPE values should be preserved after processing",
        )


class TestAuxiliaryColumnErrorHandling:
    """Property-based tests for auxiliary column error handling."""

    @given(
        df=valid_saemix_dataframe(
            min_subjects=2,
            max_subjects=10,
            min_obs_per_subject=3,
            max_obs_per_subject=5,
            include_mdv=False,
            include_cens=False,
            include_occ=False,
            include_ytype=False,
        ),
        missing_col_type=st.sampled_from(["mdv", "cens", "occ", "ytype"]),
    )
    @settings(max_examples=100)
    def test_auxiliary_column_error_handling(self, df, missing_col_type):
        """
        Feature: saemix-robustness-optimization
        Property 2: Auxiliary Column Error Handling
        Validates: Requirements 1.2

        For any column name specification where the specified column does not
        exist in the DataFrame, SaemixData SHALL raise a ValueError containing
        the missing column name.
        """
        # Generate a non-existent column name
        nonexistent_col = f"NONEXISTENT_{missing_col_type.upper()}_COL"

        # Ensure the column doesn't exist
        assume(nonexistent_col not in df.columns)

        # Build kwargs based on which column type we're testing
        kwargs = {
            "name_data": df,
            "name_group": "ID",
            "name_predictors": ["TIME"],
            "name_response": "DV",
            "verbose": False,
        }

        # Set the missing column specification
        kwargs[f"name_{missing_col_type}"] = nonexistent_col

        # Should raise ValueError with the missing column name
        with pytest.raises(ValueError) as exc_info:
            SaemixData(**kwargs)

        # Verify error message contains the missing column name
        error_message = str(exc_info.value)
        assert nonexistent_col in error_message, (
            f"Error message should contain the missing column name '{nonexistent_col}'. "
            f"Got: {error_message}"
        )
        assert missing_col_type in error_message.lower(), (
            f"Error message should indicate the column type '{missing_col_type}'. "
            f"Got: {error_message}"
        )


class TestCovariateValidationProperties:
    """Property-based tests for covariate validation correctness."""

    @given(
        df=valid_saemix_dataframe(
            min_subjects=2,
            max_subjects=10,
            min_obs_per_subject=3,
            max_obs_per_subject=5,
            include_mdv=False,
            include_cens=False,
            include_occ=False,
            include_ytype=False,
        ),
        covariate_info=st.tuples(
            st.lists(
                st.sampled_from(["AGE", "WT", "SEX", "RACE", "HT"]),
                min_size=1,
                max_size=3,
                unique=True,
            ),
            st.lists(
                st.sampled_from(["MISSING1", "MISSING2", "NOTFOUND"]),
                min_size=0,
                max_size=2,
                unique=True,
            ),
        ),
    )
    @settings(max_examples=100)
    def test_covariate_validation_correctness(self, df, covariate_info):
        """
        Feature: saemix-robustness-optimization
        Property 4: Covariate Validation Correctness
        Validates: Requirements 2.1, 2.2

        For any list of covariate names where some exist in the DataFrame and
        some do not, after validation the covariate list SHALL contain exactly
        the covariates that exist in the DataFrame, regardless of their original
        position in the input list.
        """
        valid_cov_names, invalid_cov_names = covariate_info
        df = df.copy()

        # Add valid covariate columns to the DataFrame
        for cov in valid_cov_names:
            if cov == "AGE":
                df[cov] = np.random.randint(20, 80, size=len(df))
            elif cov == "WT":
                df[cov] = np.random.uniform(50, 100, size=len(df))
            elif cov == "SEX":
                df[cov] = np.random.randint(0, 2, size=len(df))
            elif cov == "RACE":
                df[cov] = np.random.randint(1, 4, size=len(df))
            elif cov == "HT":
                df[cov] = np.random.uniform(150, 200, size=len(df))

        # Combine valid and invalid covariate names in random order
        all_covariates = list(valid_cov_names) + list(invalid_cov_names)
        np.random.shuffle(all_covariates)

        # Create SaemixData with mixed covariate list
        saemix_data = SaemixData(
            name_data=df,
            name_group="ID",
            name_predictors=["TIME"],
            name_response="DV",
            name_covariates=all_covariates,
            verbose=False,
        )

        # Verify: the resulting covariate list should contain exactly the valid covariates
        result_covariates = set(saemix_data.name_covariates)
        expected_covariates = set(valid_cov_names)

        assert result_covariates == expected_covariates, (
            f"Covariate list should contain exactly the valid covariates. "
            f"Expected: {expected_covariates}, Got: {result_covariates}"
        )

        # Verify: no invalid covariates should be in the result
        for invalid_cov in invalid_cov_names:
            assert (
                invalid_cov not in saemix_data.name_covariates
            ), f"Invalid covariate '{invalid_cov}' should not be in the result list"

    @given(
        df=valid_saemix_dataframe(
            min_subjects=2,
            max_subjects=10,
            min_obs_per_subject=3,
            max_obs_per_subject=5,
            include_mdv=False,
            include_cens=False,
            include_occ=False,
            include_ytype=False,
        ),
    )
    @settings(max_examples=100)
    def test_all_covariates_missing_results_empty_list(self, df):
        """
        Feature: saemix-robustness-optimization
        Property 4 (edge case): All Covariates Missing
        Validates: Requirements 2.3

        When all specified covariates are missing from the DataFrame,
        the covariate list SHALL be set to empty.
        """
        # Specify only non-existent covariates
        invalid_covariates = ["MISSING1", "MISSING2", "NOTFOUND"]

        # Create SaemixData with all invalid covariates
        saemix_data = SaemixData(
            name_data=df,
            name_group="ID",
            name_predictors=["TIME"],
            name_response="DV",
            name_covariates=invalid_covariates,
            verbose=False,
        )

        # Verify: the covariate list should be empty
        assert saemix_data.name_covariates == [], (
            f"Covariate list should be empty when all covariates are missing. "
            f"Got: {saemix_data.name_covariates}"
        )

    @given(
        df=valid_saemix_dataframe(
            min_subjects=2,
            max_subjects=10,
            min_obs_per_subject=3,
            max_obs_per_subject=5,
            include_mdv=False,
            include_cens=False,
            include_occ=False,
            include_ytype=False,
        ),
        valid_cov_names=st.lists(
            st.sampled_from(["AGE", "WT", "SEX"]),
            min_size=1,
            max_size=3,
            unique=True,
        ),
    )
    @settings(max_examples=100)
    def test_covariate_list_stability(self, df, valid_cov_names):
        """
        Feature: saemix-robustness-optimization
        Property 4 (stability): Covariate List Stability
        Validates: Requirements 2.5

        When covariate validation completes, the covariate list SHALL be
        stable and predictable (same input produces same output).
        """
        df = df.copy()

        # Add valid covariate columns to the DataFrame
        for cov in valid_cov_names:
            if cov == "AGE":
                df[cov] = np.random.randint(20, 80, size=len(df))
            elif cov == "WT":
                df[cov] = np.random.uniform(50, 100, size=len(df))
            elif cov == "SEX":
                df[cov] = np.random.randint(0, 2, size=len(df))

        # Create SaemixData twice with the same input
        saemix_data1 = SaemixData(
            name_data=df.copy(),
            name_group="ID",
            name_predictors=["TIME"],
            name_response="DV",
            name_covariates=list(valid_cov_names),
            verbose=False,
        )

        saemix_data2 = SaemixData(
            name_data=df.copy(),
            name_group="ID",
            name_predictors=["TIME"],
            name_response="DV",
            name_covariates=list(valid_cov_names),
            verbose=False,
        )

        # Verify: both should have the same covariate list
        assert saemix_data1.name_covariates == saemix_data2.name_covariates, (
            f"Covariate list should be stable across multiple instantiations. "
            f"First: {saemix_data1.name_covariates}, Second: {saemix_data2.name_covariates}"
        )
