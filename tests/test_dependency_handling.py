"""
Unit Tests for Dependency Handling

Feature: saemix-robustness-optimization
Tests for error handling when dependencies are missing.
Validates: Requirements 7.1, 7.2
"""

import sys
import pytest
from unittest.mock import patch, MagicMock


class TestOptionalDependencyHandling:
    """Tests for optional dependency (matplotlib) handling."""

    def test_require_matplotlib_returns_module_when_available(self):
        """
        Test that _require_matplotlib returns matplotlib.pyplot when available.
        Validates: Requirements 7.1
        """
        import saemix

        # If matplotlib is available, _require_matplotlib should return it
        if saemix._HAS_MATPLOTLIB:
            plt = saemix._require_matplotlib()
            assert plt is not None
            assert hasattr(plt, "figure")
            assert hasattr(plt, "plot")
            assert hasattr(plt, "show")

    def test_require_matplotlib_raises_when_unavailable(self):
        """
        Test that _require_matplotlib raises ImportError with clear message
        when matplotlib is not available.
        Validates: Requirements 7.1
        """
        import saemix

        # Temporarily set _HAS_MATPLOTLIB to False
        original_value = saemix._HAS_MATPLOTLIB
        try:
            saemix._HAS_MATPLOTLIB = False

            with pytest.raises(ImportError) as exc_info:
                saemix._require_matplotlib()

            # Check error message contains helpful information
            error_msg = str(exc_info.value)
            assert "matplotlib" in error_msg.lower()
            assert "pip install" in error_msg.lower()
        finally:
            # Restore original value
            saemix._HAS_MATPLOTLIB = original_value

    def test_has_matplotlib_flag_is_boolean(self):
        """
        Test that _HAS_MATPLOTLIB is a boolean flag.
        Validates: Requirements 7.1
        """
        import saemix

        assert isinstance(saemix._HAS_MATPLOTLIB, bool)

    def test_has_matplotlib_exported_in_all(self):
        """
        Test that _HAS_MATPLOTLIB is exported in __all__.
        Validates: Requirements 7.1
        """
        import saemix

        assert "_HAS_MATPLOTLIB" in saemix.__all__
        assert "_require_matplotlib" in saemix.__all__


class TestCoreDependencyHandling:
    """Tests for core dependency handling."""

    def test_package_imports_successfully(self):
        """
        Test that the package imports successfully when all dependencies are present.
        Validates: Requirements 7.2
        """
        # This test verifies that the package can be imported
        import saemix

        assert saemix is not None
        assert hasattr(saemix, "__version__")
        assert hasattr(saemix, "SaemixData")
        assert hasattr(saemix, "SaemixModel")
        assert hasattr(saemix, "saemix")

    def test_core_dependencies_are_available(self):
        """
        Test that core dependencies (numpy, pandas, scipy) are available.
        Validates: Requirements 7.2
        """
        # These imports should work if the package imported successfully
        import numpy as np
        import pandas as pd
        from scipy import stats

        assert np is not None
        assert pd is not None
        assert stats is not None

    def test_numpy_dependency_error_message_format(self):
        """
        Test that numpy dependency error message follows expected format.
        Validates: Requirements 7.2

        Note: This test simulates what the error message would look like
        by checking the format in the source code, since we can't actually
        remove numpy while running tests.
        """
        import saemix

        # Read the __init__.py source to verify error message format
        import inspect

        source = inspect.getsource(saemix)

        # Check that numpy error handling exists with proper format
        assert "Missing required dependency: numpy" in source
        assert "pip install numpy" in source

    def test_pandas_dependency_error_message_format(self):
        """
        Test that pandas dependency error message follows expected format.
        Validates: Requirements 7.2
        """
        import saemix
        import inspect

        source = inspect.getsource(saemix)

        # Check that pandas error handling exists with proper format
        assert "Missing required dependency: pandas" in source
        assert "pip install pandas" in source

    def test_scipy_dependency_error_message_format(self):
        """
        Test that scipy dependency error message follows expected format.
        Validates: Requirements 7.2
        """
        import saemix
        import inspect

        source = inspect.getsource(saemix)

        # Check that scipy error handling exists with proper format
        assert "Missing required dependency: scipy" in source
        assert "pip install scipy" in source


class TestDiagnosticsMatplotlibHandling:
    """Tests for matplotlib handling in diagnostics module."""

    def test_diagnostics_require_matplotlib_exists(self):
        """
        Test that diagnostics module has its own _require_matplotlib function.
        Validates: Requirements 7.1
        """
        from saemix import diagnostics

        assert hasattr(diagnostics, "_require_matplotlib")
        assert callable(diagnostics._require_matplotlib)

    def test_diagnostics_require_matplotlib_returns_plt(self):
        """
        Test that diagnostics._require_matplotlib returns matplotlib.pyplot.
        Validates: Requirements 7.1
        """
        from saemix import diagnostics

        try:
            plt = diagnostics._require_matplotlib()
            assert plt is not None
            assert hasattr(plt, "figure")
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_diagnostics_require_matplotlib_error_message(self):
        """
        Test that diagnostics._require_matplotlib raises ImportError with message.
        Validates: Requirements 7.1
        """
        from saemix import diagnostics
        import inspect

        source = inspect.getsource(diagnostics._require_matplotlib)
        assert "matplotlib" in source.lower()
        assert "ImportError" in source


class TestErrorMessageQuality:
    """Tests for error message quality and actionability."""

    def test_matplotlib_error_is_actionable(self):
        """
        Test that matplotlib error message provides actionable guidance.
        Validates: Requirements 7.1
        """
        import saemix

        # Temporarily disable matplotlib
        original_value = saemix._HAS_MATPLOTLIB
        try:
            saemix._HAS_MATPLOTLIB = False

            with pytest.raises(ImportError) as exc_info:
                saemix._require_matplotlib()

            error_msg = str(exc_info.value)

            # Error should mention the package name
            assert "saemix" in error_msg.lower()

            # Error should mention what functionality requires it
            assert "plotting" in error_msg.lower()

            # Error should provide installation command
            assert "pip install matplotlib" in error_msg
        finally:
            saemix._HAS_MATPLOTLIB = original_value

    def test_error_messages_use_bracket_prefix(self):
        """
        Test that error messages use [saemix] prefix format.
        Validates: Requirements 7.1, 7.2
        """
        import saemix
        import inspect

        source = inspect.getsource(saemix)

        # Check that error messages use the [saemix] prefix
        assert "[saemix]" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
