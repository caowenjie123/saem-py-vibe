"""
Property-Based Tests for Package Structure

Feature: pypi-package
Property 3: 包结构完整性 (Package Structure Integrity)
Validates: Requirements 4.1, 4.2, 4.4
"""

import importlib
import os
import sys
import pytest
from hypothesis import given, settings, strategies as st


class TestPackageStructure:
    """Tests for package structure integrity."""

    def test_version_attribute_exists(self):
        """
        Test that __version__ attribute exists and is properly formatted.
        Validates: Requirements 2.1, 2.3
        """
        import saemix

        # Check __version__ exists
        assert hasattr(
            saemix, "__version__"
        ), "Package should have __version__ attribute"

        # Check version is a string
        assert isinstance(saemix.__version__, str), "__version__ should be a string"

        # Check version follows semantic versioning (MAJOR.MINOR.PATCH)
        version_parts = saemix.__version__.split(".")
        assert (
            len(version_parts) >= 2
        ), "Version should have at least MAJOR.MINOR format"

        # Check all parts are numeric
        for part in version_parts:
            assert part.isdigit(), f"Version part '{part}' should be numeric"

    def test_version_info_attribute_exists(self):
        """
        Test that __version_info__ attribute exists and is a tuple.
        Validates: Requirements 2.1
        """
        import saemix

        assert hasattr(
            saemix, "__version_info__"
        ), "Package should have __version_info__ attribute"
        assert isinstance(
            saemix.__version_info__, tuple
        ), "__version_info__ should be a tuple"

        # Check consistency with __version__
        expected_info = tuple(int(x) for x in saemix.__version__.split("."))
        assert (
            saemix.__version_info__ == expected_info
        ), "__version_info__ should match __version__"

    def test_init_files_exist(self):
        """
        Test that all required __init__.py files exist.
        Validates: Requirements 4.2
        """
        import saemix

        # Get package directory
        package_dir = os.path.dirname(saemix.__file__)

        # Check main package __init__.py
        main_init = os.path.join(package_dir, "__init__.py")
        assert os.path.exists(main_init), "saemix/__init__.py should exist"

        # Check algorithm subpackage __init__.py
        algorithm_init = os.path.join(package_dir, "algorithm", "__init__.py")
        assert os.path.exists(
            algorithm_init
        ), "saemix/algorithm/__init__.py should exist"

        # Check _version.py exists
        version_file = os.path.join(package_dir, "_version.py")
        assert os.path.exists(version_file), "saemix/_version.py should exist"

    def test_core_modules_importable(self):
        """
        Test that all core modules are importable.
        Validates: Requirements 4.1, 4.4
        """
        core_modules = [
            "saemix",
            "saemix.data",
            "saemix.model",
            "saemix.control",
            "saemix.results",
            "saemix.main",
            "saemix.utils",
            "saemix.export",
            "saemix.diagnostics",
            "saemix.simulation",
            "saemix.compare",
            "saemix.stepwise",
            "saemix.plot_options",
            "saemix._version",
        ]

        for module_name in core_modules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None, f"Module {module_name} should be importable"
            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")

    def test_algorithm_subpackage_importable(self):
        """
        Test that algorithm subpackage modules are importable.
        Validates: Requirements 4.1, 4.4
        """
        algorithm_modules = [
            "saemix.algorithm",
            "saemix.algorithm.saem",
            "saemix.algorithm.estep",
            "saemix.algorithm.mstep",
            "saemix.algorithm.initialization",
            "saemix.algorithm.conddist",
            "saemix.algorithm.likelihood",
            "saemix.algorithm.map_estimation",
            "saemix.algorithm.fim",
            "saemix.algorithm.predict",
        ]

        for module_name in algorithm_modules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None, f"Module {module_name} should be importable"
            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")

    def test_public_api_exports(self):
        """
        Test that public API classes and functions are exported from main package.
        Validates: Requirements 4.1
        """
        import saemix

        # Core classes that should be exported
        expected_classes = [
            "SaemixData",
            "SaemixModel",
            "SaemixObject",
            "SaemixRes",
            "PlotOptions",
        ]

        # Core functions that should be exported
        expected_functions = [
            "saemix",
            "saemix_data",
            "saemix_model",
            "saemix_control",
            "conddist_saemix",
            "compare_saemix",
            "simulate_saemix",
            "save_results",
            "export_to_csv",
        ]

        for name in expected_classes:
            assert hasattr(saemix, name), f"Class {name} should be exported from saemix"

        for name in expected_functions:
            assert hasattr(
                saemix, name
            ), f"Function {name} should be exported from saemix"

    def test_all_attribute_defined(self):
        """
        Test that __all__ is defined and contains expected exports.
        Validates: Requirements 4.1
        """
        import saemix

        assert hasattr(saemix, "__all__"), "Package should define __all__"
        assert isinstance(saemix.__all__, list), "__all__ should be a list"
        assert len(saemix.__all__) > 0, "__all__ should not be empty"

        # Check that __version__ is in __all__
        assert "__version__" in saemix.__all__, "__version__ should be in __all__"


class TestPackageStructureProperty:
    """Property-based tests for package structure integrity."""

    @settings(max_examples=100, deadline=None)
    @given(module_index=st.integers(min_value=0, max_value=13))
    def test_property_3_package_structure_integrity(self, module_index):
        """
        Feature: pypi-package, Property 3: 包结构完整性
        Validates: Requirements 4.1, 4.2, 4.4

        For any built distribution (sdist or wheel), all Python modules under
        saemix/ directory SHALL be included and importable.

        This test verifies that for any randomly selected module from the package,
        the module can be successfully imported and has the expected attributes.
        """
        all_modules = [
            "saemix",
            "saemix.data",
            "saemix.model",
            "saemix.control",
            "saemix.results",
            "saemix.main",
            "saemix.utils",
            "saemix.export",
            "saemix.diagnostics",
            "saemix.simulation",
            "saemix.compare",
            "saemix.stepwise",
            "saemix.plot_options",
            "saemix._version",
        ]

        # Select module based on index
        module_name = all_modules[module_index % len(all_modules)]

        # Import the module
        module = importlib.import_module(module_name)

        # Verify module is not None
        assert module is not None, f"Module {module_name} should be importable"

        # Verify module has __name__ attribute
        assert hasattr(module, "__name__"), f"Module {module_name} should have __name__"
        assert (
            module.__name__ == module_name
        ), f"Module __name__ should match import name"

        # Verify module has __file__ attribute (indicates it's a real file)
        assert hasattr(module, "__file__"), f"Module {module_name} should have __file__"
        assert (
            module.__file__ is not None
        ), f"Module {module_name}.__file__ should not be None"

        # Verify the file exists
        assert os.path.exists(
            module.__file__
        ), f"Module file {module.__file__} should exist"

    @settings(max_examples=100, deadline=None)
    @given(algo_index=st.integers(min_value=0, max_value=9))
    def test_property_3_algorithm_subpackage_integrity(self, algo_index):
        """
        Feature: pypi-package, Property 3: 包结构完整性
        Validates: Requirements 4.1, 4.2, 4.4

        For any module in the algorithm subpackage, the module SHALL be
        importable and have valid file references.
        """
        algorithm_modules = [
            "saemix.algorithm",
            "saemix.algorithm.saem",
            "saemix.algorithm.estep",
            "saemix.algorithm.mstep",
            "saemix.algorithm.initialization",
            "saemix.algorithm.conddist",
            "saemix.algorithm.likelihood",
            "saemix.algorithm.map_estimation",
            "saemix.algorithm.fim",
            "saemix.algorithm.predict",
        ]

        # Select module based on index
        module_name = algorithm_modules[algo_index % len(algorithm_modules)]

        # Import the module
        module = importlib.import_module(module_name)

        # Verify module is not None
        assert module is not None, f"Module {module_name} should be importable"

        # Verify module has __name__ attribute
        assert hasattr(module, "__name__"), f"Module {module_name} should have __name__"

        # Verify module has __file__ attribute
        assert hasattr(module, "__file__"), f"Module {module_name} should have __file__"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
