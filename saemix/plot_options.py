"""
Plot Options Module

This module provides a centralized system for managing plot appearance settings.
Implements Requirements 8.1-8.5.
"""

from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, Dict, Any
import copy


@dataclass
class PlotOptions:
    """
    Configuration class for plot appearance settings.

    This class stores default values for all plot parameters and provides
    methods to apply these settings to matplotlib.

    Attributes
    ----------
    figsize : tuple of int
        Figure size (width, height) in inches
    dpi : int
        Resolution in dots per inch
    style : str
        Matplotlib style name
    color_palette : str
        Color palette name (e.g., 'tab10', 'Set1')
    alpha : float
        Default transparency for scatter points
    marker_size : int
        Default marker size for scatter plots
    line_width : float
        Default line width
    font_size : int
        Default font size for labels
    title_size : int
        Font size for titles
    legend_fontsize : int
        Font size for legends
    grid : bool
        Whether to show grid lines
    grid_alpha : float
        Transparency of grid lines
    spine_visible : bool
        Whether to show axis spines
    tight_layout : bool
        Whether to use tight layout
    """

    figsize: Tuple[int, int] = (10, 8)
    dpi: int = 100
    style: str = "seaborn-v0_8-whitegrid"
    color_palette: str = "tab10"
    alpha: float = 0.7
    marker_size: int = 20
    line_width: float = 1.5
    font_size: int = 12
    title_size: int = 14
    legend_fontsize: int = 10
    grid: bool = True
    grid_alpha: float = 0.3
    spine_visible: bool = True
    tight_layout: bool = True

    def apply(self) -> None:
        """
        Apply current options to matplotlib's rcParams.

        This method updates matplotlib's global settings to match
        the current PlotOptions configuration.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib as mpl
        except ImportError:
            raise ImportError("matplotlib is required for plot options")

        # Try to set style, fall back to default if not available
        try:
            plt.style.use(self.style)
        except OSError:
            # Style not available, use default
            try:
                plt.style.use("seaborn-v0_8-whitegrid")
            except OSError:
                try:
                    plt.style.use("seaborn-whitegrid")
                except OSError:
                    pass  # Use default style

        # Set rcParams
        mpl.rcParams["figure.figsize"] = self.figsize
        mpl.rcParams["figure.dpi"] = self.dpi
        mpl.rcParams["font.size"] = self.font_size
        mpl.rcParams["axes.titlesize"] = self.title_size
        mpl.rcParams["legend.fontsize"] = self.legend_fontsize
        mpl.rcParams["lines.linewidth"] = self.line_width
        mpl.rcParams["lines.markersize"] = self.marker_size**0.5  # Convert to radius
        mpl.rcParams["axes.grid"] = self.grid
        mpl.rcParams["grid.alpha"] = self.grid_alpha

        # Set color cycle from palette
        try:
            from matplotlib import cm

            if hasattr(cm, self.color_palette):
                cmap = getattr(cm, self.color_palette)
                if hasattr(cmap, "colors"):
                    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=cmap.colors)
        except Exception:
            pass  # Keep default color cycle

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert options to dictionary.

        Returns
        -------
        dict
            Dictionary of all option values
        """
        return asdict(self)

    def copy(self) -> "PlotOptions":
        """
        Create a copy of the current options.

        Returns
        -------
        PlotOptions
            A new PlotOptions instance with the same values
        """
        return copy.deepcopy(self)

    def update(self, **kwargs) -> None:
        """
        Update options with provided keyword arguments.

        Parameters
        ----------
        **kwargs
            Option names and values to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown option: {key}")


# =============================================================================
# Global Options Management
# =============================================================================

# Store default options for reset
_DEFAULT_OPTIONS = PlotOptions()

# Global options instance
_plot_options = PlotOptions()


def set_plot_options(**kwargs) -> None:
    """
    Set global plot options.

    Updates the global default options that will be used by all
    diagnostic plotting functions.

    Parameters
    ----------
    **kwargs
        Option names and values to set. Valid options include:
        - figsize: tuple of (width, height) in inches
        - dpi: int, resolution
        - style: str, matplotlib style name
        - color_palette: str, color palette name
        - alpha: float, transparency (0-1)
        - marker_size: int, marker size
        - line_width: float, line width
        - font_size: int, font size
        - title_size: int, title font size
        - legend_fontsize: int, legend font size
        - grid: bool, show grid
        - grid_alpha: float, grid transparency
        - spine_visible: bool, show spines
        - tight_layout: bool, use tight layout

    Examples
    --------
    >>> set_plot_options(figsize=(12, 8), dpi=150)
    >>> set_plot_options(alpha=0.5, marker_size=30)
    """
    global _plot_options
    _plot_options.update(**kwargs)


def get_plot_options() -> PlotOptions:
    """
    Get current global plot options.

    Returns
    -------
    PlotOptions
        The current global plot options instance

    Examples
    --------
    >>> opts = get_plot_options()
    >>> print(opts.figsize)
    (10, 8)
    """
    global _plot_options
    return _plot_options


def reset_plot_options() -> None:
    """
    Reset all plot options to their default values.

    This restores all options to the original defaults defined
    in the PlotOptions class.

    Examples
    --------
    >>> set_plot_options(figsize=(20, 16))
    >>> reset_plot_options()
    >>> get_plot_options().figsize
    (10, 8)
    """
    global _plot_options, _DEFAULT_OPTIONS
    _plot_options = PlotOptions()


def apply_plot_options() -> None:
    """
    Apply current global options to matplotlib.

    This updates matplotlib's rcParams to match the current
    global plot options.

    Examples
    --------
    >>> set_plot_options(figsize=(12, 8))
    >>> apply_plot_options()  # Now matplotlib will use (12, 8) as default
    """
    global _plot_options
    _plot_options.apply()


def with_plot_options(func):
    """
    Decorator to apply plot options before calling a plotting function.

    This decorator ensures that the global plot options are applied
    to matplotlib before the decorated function is called.

    Parameters
    ----------
    func : callable
        The plotting function to decorate

    Returns
    -------
    callable
        The decorated function
    """

    def wrapper(*args, **kwargs):
        apply_plot_options()
        return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


def merge_options(local_options: Optional[Dict[str, Any]] = None) -> PlotOptions:
    """
    Merge local options with global defaults.

    Local options override global defaults when both are specified.

    Parameters
    ----------
    local_options : dict, optional
        Local option overrides

    Returns
    -------
    PlotOptions
        Merged options instance

    Examples
    --------
    >>> opts = merge_options({'figsize': (15, 10)})
    >>> opts.figsize
    (15, 10)
    """
    global _plot_options
    merged = _plot_options.copy()
    if local_options:
        merged.update(**local_options)
    return merged


def get_figsize(local_figsize: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    """
    Get figure size, preferring local value over global default.

    Parameters
    ----------
    local_figsize : tuple, optional
        Local figure size override

    Returns
    -------
    tuple
        Figure size (width, height)
    """
    if local_figsize is not None:
        return local_figsize
    return get_plot_options().figsize


def get_alpha(local_alpha: Optional[float] = None) -> float:
    """
    Get alpha value, preferring local value over global default.

    Parameters
    ----------
    local_alpha : float, optional
        Local alpha override

    Returns
    -------
    float
        Alpha value
    """
    if local_alpha is not None:
        return local_alpha
    return get_plot_options().alpha
