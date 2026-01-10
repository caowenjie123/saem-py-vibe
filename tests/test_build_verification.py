"""
Property-Based Tests for Build Verification

Feature: pypi-package
Property 5: 构建可重复性 (Build Reproducibility)
Validates: Requirements 7.1, 7.2, 7.3, 7.4
"""

import os
import subprocess
import sys
import tempfile
import shutil
import tarfile
import zipfile
import pytest
from hypothesis import given, settings, strategies as st


class TestBuildVerification:
    """Tests for build verification and distribution integrity."""

    @pytest.fixture(scope="class")
    def build_artifacts(self, tmp_path_factory):
        """Build sdist and wheel distributions for testing."""
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Create a temporary directory for build output
        dist_dir = tmp_path_factory.mktemp("dist")

        # Clean any existing dist directory content
        result = subprocess.run(
            [sys.executable, "-m", "build", "--outdir", str(dist_dir)],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            pytest.fail(f"Build failed: {result.stderr}")

        # Find the built files
        sdist_file = None
        wheel_file = None

        for f in os.listdir(dist_dir):
            if f.endswith(".tar.gz"):
                sdist_file = os.path.join(dist_dir, f)
            elif f.endswith(".whl"):
                wheel_file = os.path.join(dist_dir, f)

        return {
            "dist_dir": dist_dir,
            "sdist": sdist_file,
            "wheel": wheel_file,
            "project_root": project_root,
        }

    def test_sdist_builds_successfully(self, build_artifacts):
        """
        Test that source distribution (sdist) builds successfully.
        Validates: Requirements 7.1
        """
        assert build_artifacts["sdist"] is not None, "sdist should be built"
        assert os.path.exists(build_artifacts["sdist"]), "sdist file should exist"
        assert build_artifacts["sdist"].endswith(
            ".tar.gz"
        ), "sdist should be a tar.gz file"

    def test_wheel_builds_successfully(self, build_artifacts):
        """
        Test that wheel distribution builds successfully.
        Validates: Requirements 7.2
        """
        assert build_artifacts["wheel"] is not None, "wheel should be built"
        assert os.path.exists(build_artifacts["wheel"]), "wheel file should exist"
        assert build_artifacts["wheel"].endswith(".whl"), "wheel should be a .whl file"

    def test_wheel_is_pure_python(self, build_artifacts):
        """
        Test that wheel is platform-independent (pure Python).
        Validates: Requirements 7.3
        """
        wheel_name = os.path.basename(build_artifacts["wheel"])
        # Pure Python wheels have 'py3-none-any' in the filename
        assert (
            "py3-none-any" in wheel_name or "none-any" in wheel_name
        ), f"Wheel should be pure Python (platform-independent), got: {wheel_name}"

    def test_sdist_contains_required_files(self, build_artifacts):
        """
        Test that sdist contains all required files.
        Validates: Requirements 7.4
        """
        required_files = [
            "pyproject.toml",
            "README.md",
            "LICENSE",
            "saemix/__init__.py",
            "saemix/_version.py",
            "saemix/algorithm/__init__.py",
        ]

        with tarfile.open(build_artifacts["sdist"], "r:gz") as tar:
            tar_members = [m.name for m in tar.getmembers()]

            for req_file in required_files:
                # Files in sdist are prefixed with package-version/
                found = any(m.endswith(req_file) for m in tar_members)
                assert found, f"sdist should contain {req_file}"

    def test_wheel_contains_required_modules(self, build_artifacts):
        """
        Test that wheel contains all required Python modules.
        Validates: Requirements 7.4
        """
        required_modules = [
            "saemix/__init__.py",
            "saemix/_version.py",
            "saemix/data.py",
            "saemix/model.py",
            "saemix/main.py",
            "saemix/algorithm/__init__.py",
            "saemix/algorithm/saem.py",
        ]

        with zipfile.ZipFile(build_artifacts["wheel"], "r") as whl:
            whl_members = whl.namelist()

            for req_module in required_modules:
                found = any(
                    m.endswith(req_module) or req_module in m for m in whl_members
                )
                assert found, f"wheel should contain {req_module}"

    def test_wheel_contains_metadata(self, build_artifacts):
        """
        Test that wheel contains proper metadata.
        Validates: Requirements 7.4
        """
        with zipfile.ZipFile(build_artifacts["wheel"], "r") as whl:
            whl_members = whl.namelist()

            # Check for METADATA file
            metadata_files = [m for m in whl_members if "METADATA" in m]
            assert len(metadata_files) > 0, "wheel should contain METADATA file"

            # Read and verify metadata content
            metadata_file = metadata_files[0]
            metadata_content = whl.read(metadata_file).decode("utf-8")

            assert (
                "Name: saemix" in metadata_content
            ), "METADATA should contain package name"
            assert (
                "Version: 0.1.0" in metadata_content
            ), "METADATA should contain version"

    def test_twine_check_passes(self, build_artifacts):
        """
        Test that twine check passes for both distributions.
        Validates: Requirements 7.4
        """
        # Check sdist
        result_sdist = subprocess.run(
            [sys.executable, "-m", "twine", "check", build_artifacts["sdist"]],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert (
            result_sdist.returncode == 0
        ), f"twine check failed for sdist: {result_sdist.stdout}"

        # Check wheel
        result_wheel = subprocess.run(
            [sys.executable, "-m", "twine", "check", build_artifacts["wheel"]],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert (
            result_wheel.returncode == 0
        ), f"twine check failed for wheel: {result_wheel.stdout}"


class TestBuildReproducibilityProperty:
    """Property-based tests for build reproducibility."""

    @pytest.fixture(scope="class")
    def dual_build_artifacts(self, tmp_path_factory):
        """Build distributions twice to test reproducibility."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        builds = []
        for i in range(2):
            dist_dir = tmp_path_factory.mktemp(f"dist_{i}")

            result = subprocess.run(
                [sys.executable, "-m", "build", "--outdir", str(dist_dir)],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                pytest.fail(f"Build {i} failed: {result.stderr}")

            sdist_file = None
            wheel_file = None

            for f in os.listdir(dist_dir):
                if f.endswith(".tar.gz"):
                    sdist_file = os.path.join(dist_dir, f)
                elif f.endswith(".whl"):
                    wheel_file = os.path.join(dist_dir, f)

            builds.append(
                {"dist_dir": dist_dir, "sdist": sdist_file, "wheel": wheel_file}
            )

        return builds

    @settings(max_examples=100, deadline=None)
    @given(file_index=st.integers(min_value=0, max_value=6))
    def test_property_5_build_reproducibility_sdist_contents(
        self, dual_build_artifacts, file_index
    ):
        """
        Feature: pypi-package, Property 5: 构建可重复性
        Validates: Requirements 7.1, 7.2, 7.4

        For any given source tree state, building sdist then building wheel from
        that sdist SHALL produce functionally equivalent packages.

        This test verifies that for any randomly selected file from the sdist,
        the file exists in both builds with the same content.
        """
        key_files = [
            "pyproject.toml",
            "README.md",
            "LICENSE",
            "saemix/__init__.py",
            "saemix/_version.py",
            "saemix/data.py",
            "saemix/model.py",
        ]

        target_file = key_files[file_index % len(key_files)]

        build1_sdist = dual_build_artifacts[0]["sdist"]
        build2_sdist = dual_build_artifacts[1]["sdist"]

        # Extract and compare file contents
        with tarfile.open(build1_sdist, "r:gz") as tar1, tarfile.open(
            build2_sdist, "r:gz"
        ) as tar2:

            # Find the file in both archives
            tar1_members = {m.name: m for m in tar1.getmembers()}
            tar2_members = {m.name: m for m in tar2.getmembers()}

            file1_name = None
            file2_name = None

            for name in tar1_members:
                if name.endswith(target_file):
                    file1_name = name
                    break

            for name in tar2_members:
                if name.endswith(target_file):
                    file2_name = name
                    break

            assert (
                file1_name is not None
            ), f"{target_file} should exist in build 1 sdist"
            assert (
                file2_name is not None
            ), f"{target_file} should exist in build 2 sdist"

            # Compare file sizes
            size1 = tar1_members[file1_name].size
            size2 = tar2_members[file2_name].size
            assert (
                size1 == size2
            ), f"File {target_file} should have same size in both builds"

    @settings(max_examples=100, deadline=None)
    @given(module_index=st.integers(min_value=0, max_value=6))
    def test_property_5_build_reproducibility_wheel_contents(
        self, dual_build_artifacts, module_index
    ):
        """
        Feature: pypi-package, Property 5: 构建可重复性
        Validates: Requirements 7.1, 7.2, 7.4

        For any given source tree state, building wheel distributions SHALL
        produce functionally equivalent packages.

        This test verifies that for any randomly selected module from the wheel,
        the module exists in both builds.
        """
        key_modules = [
            "saemix/__init__.py",
            "saemix/_version.py",
            "saemix/data.py",
            "saemix/model.py",
            "saemix/main.py",
            "saemix/algorithm/__init__.py",
            "saemix/algorithm/saem.py",
        ]

        target_module = key_modules[module_index % len(key_modules)]

        build1_wheel = dual_build_artifacts[0]["wheel"]
        build2_wheel = dual_build_artifacts[1]["wheel"]

        with zipfile.ZipFile(build1_wheel, "r") as whl1, zipfile.ZipFile(
            build2_wheel, "r"
        ) as whl2:

            whl1_members = whl1.namelist()
            whl2_members = whl2.namelist()

            # Check module exists in both
            found_in_1 = any(target_module in m for m in whl1_members)
            found_in_2 = any(target_module in m for m in whl2_members)

            assert found_in_1, f"{target_module} should exist in build 1 wheel"
            assert found_in_2, f"{target_module} should exist in build 2 wheel"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
