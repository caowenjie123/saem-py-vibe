"""
Property-Based Tests for Plot Options Module

Feature: saemix-python-enhancement
Property 12: Plot Options Management
Validates: Requirements 8.2, 8.3, 8.4, 8.5
"""

import pytest
from hypothesis import given, settings, strategies as st

from saemix.plot_options import (
    PlotOptions,
    set_plot_options,
    get_plot_options,
    reset_plot_options,
    merge_options,
    get_figsize,
    get_alpha,
)


class TestPlotOptionsManagement:
    """Property-based tests for plot options management."""

    def setup_method(self):
        """Reset options before each test."""
        reset_plot_options()

    def teardown_method(self):
        """Reset options after each test."""
        reset_plot_options()

    @settings(max_examples=100, deadline=None)
    @given(
        width=st.integers(min_value=4, max_value=20),
        height=st.integers(min_value=4, max_value=20),
    )
    def test_property_12_set_get_figsize(self, width, height):
        """
        Feature: saemix-python-enhancement, Property 12: Plot Options Management
        Validates: Requirements 8.2, 8.3

        set_plot_options(figsize=(w, h)) followed by get_plot_options().figsize
        SHALL return (w, h).
        """
        # Set figsize
        set_plot_options(figsize=(width, height))

        # Get and verify
        opts = get_plot_options()
        assert opts.figsize == (
            width,
            height,
        ), f"Expected figsize ({width}, {height}), got {opts.figsize}"

    @settings(max_examples=100, deadline=None)
    @given(dpi=st.integers(min_value=50, max_value=300))
    def test_property_12_set_get_dpi(self, dpi):
        """
        Test that set_plot_options(dpi=x) followed by get_plot_options().dpi
        returns x.
        Validates: Requirements 8.2, 8.3
        """
        set_plot_options(dpi=dpi)
        opts = get_plot_options()
        assert opts.dpi == dpi, f"Expected dpi {dpi}, got {opts.dpi}"

    @settings(max_examples=100, deadline=None)
    @given(alpha=st.floats(min_value=0.0, max_value=1.0))
    def test_property_12_set_get_alpha(self, alpha):
        """
        Test that set_plot_options(alpha=x) followed by get_plot_options().alpha
        returns x.
        Validates: Requirements 8.2, 8.3
        """
        set_plot_options(alpha=alpha)
        opts = get_plot_options()
        assert opts.alpha == alpha, f"Expected alpha {alpha}, got {opts.alpha}"

    @settings(max_examples=100, deadline=None)
    @given(
        width=st.integers(min_value=4, max_value=20),
        height=st.integers(min_value=4, max_value=20),
        dpi=st.integers(min_value=50, max_value=300),
        alpha=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_property_12_reset_restores_defaults(self, width, height, dpi, alpha):
        """
        Feature: saemix-python-enhancement, Property 12: Plot Options Management
        Validates: Requirements 8.5

        reset_plot_options() SHALL restore all settings to their original defaults.
        """
        # Get default values
        default_opts = PlotOptions()

        # Set custom values
        set_plot_options(figsize=(width, height), dpi=dpi, alpha=alpha)

        # Verify custom values are set
        opts = get_plot_options()
        assert opts.figsize == (width, height)
        assert opts.dpi == dpi
        assert opts.alpha == alpha

        # Reset
        reset_plot_options()

        # Verify defaults are restored
        opts = get_plot_options()
        assert (
            opts.figsize == default_opts.figsize
        ), f"Expected default figsize {default_opts.figsize}, got {opts.figsize}"
        assert (
            opts.dpi == default_opts.dpi
        ), f"Expected default dpi {default_opts.dpi}, got {opts.dpi}"
        assert (
            opts.alpha == default_opts.alpha
        ), f"Expected default alpha {default_opts.alpha}, got {opts.alpha}"

    @settings(max_examples=100, deadline=None)
    @given(
        global_width=st.integers(min_value=4, max_value=20),
        global_height=st.integers(min_value=4, max_value=20),
        local_width=st.integers(min_value=4, max_value=20),
        local_height=st.integers(min_value=4, max_value=20),
    )
    def test_property_12_local_overrides_global(
        self, global_width, global_height, local_width, local_height
    ):
        """
        Feature: saemix-python-enhancement, Property 12: Plot Options Management
        Validates: Requirements 8.4

        Local options override global options when both are specified.
        """
        # Set global options
        set_plot_options(figsize=(global_width, global_height))

        # Merge with local options
        merged = merge_options({"figsize": (local_width, local_height)})

        # Local should override global
        assert merged.figsize == (local_width, local_height), (
            f"Expected local figsize ({local_width}, {local_height}), "
            f"got {merged.figsize}"
        )

    @settings(max_examples=100, deadline=None)
    @given(
        global_width=st.integers(min_value=4, max_value=20),
        global_height=st.integers(min_value=4, max_value=20),
    )
    def test_property_12_merge_preserves_global_when_no_local(
        self, global_width, global_height
    ):
        """
        Test that merge_options preserves global options when no local override.
        Validates: Requirements 8.4
        """
        # Set global options
        set_plot_options(figsize=(global_width, global_height))

        # Merge with no local options
        merged = merge_options(None)

        # Global should be preserved
        assert merged.figsize == (global_width, global_height)

    @settings(max_examples=100, deadline=None)
    @given(
        local_width=st.integers(min_value=4, max_value=20),
        local_height=st.integers(min_value=4, max_value=20),
    )
    def test_get_figsize_prefers_local(self, local_width, local_height):
        """
        Test that get_figsize prefers local value over global.
        Validates: Requirements 8.4
        """
        # Set global
        set_plot_options(figsize=(10, 8))

        # Get with local override
        result = get_figsize((local_width, local_height))

        assert result == (local_width, local_height)

    @settings(max_examples=100, deadline=None)
    @given(local_alpha=st.floats(min_value=0.0, max_value=1.0))
    def test_get_alpha_prefers_local(self, local_alpha):
        """
        Test that get_alpha prefers local value over global.
        Validates: Requirements 8.4
        """
        # Set global
        set_plot_options(alpha=0.5)

        # Get with local override
        result = get_alpha(local_alpha)

        assert result == local_alpha


class TestPlotOptionsDataclass:
    """Tests for PlotOptions dataclass."""

    def test_default_values(self):
        """Test that PlotOptions has correct default values."""
        opts = PlotOptions()

        assert opts.figsize == (10, 8)
        assert opts.dpi == 100
        assert opts.alpha == 0.7
        assert opts.marker_size == 20
        assert opts.line_width == 1.5
        assert opts.font_size == 12
        assert opts.title_size == 14
        assert opts.grid is True

    def test_to_dict(self):
        """Test that to_dict returns all options."""
        opts = PlotOptions()
        d = opts.to_dict()

        assert "figsize" in d
        assert "dpi" in d
        assert "alpha" in d
        assert d["figsize"] == (10, 8)

    def test_copy_creates_independent_instance(self):
        """Test that copy creates an independent instance."""
        opts1 = PlotOptions()
        opts2 = opts1.copy()

        # Modify opts2
        opts2.figsize = (20, 16)

        # opts1 should be unchanged
        assert opts1.figsize == (10, 8)
        assert opts2.figsize == (20, 16)

    def test_update_modifies_options(self):
        """Test that update modifies options."""
        opts = PlotOptions()
        opts.update(figsize=(15, 10), dpi=150)

        assert opts.figsize == (15, 10)
        assert opts.dpi == 150

    def test_update_invalid_option_raises_error(self):
        """Test that update with invalid option raises ValueError."""
        opts = PlotOptions()

        with pytest.raises(ValueError, match="Unknown option"):
            opts.update(invalid_option=123)


class TestPlotOptionsApply:
    """Tests for applying plot options to matplotlib."""

    def test_apply_does_not_raise(self):
        """Test that apply() doesn't raise errors."""
        opts = PlotOptions()

        # Should not raise even if matplotlib is available
        try:
            opts.apply()
        except ImportError:
            pytest.skip("matplotlib not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
