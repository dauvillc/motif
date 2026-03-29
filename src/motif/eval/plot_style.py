"""Shared publication-quality plotting style for single-column article figures.

Usage
-----
from motif.eval.plot_style import (
    apply_paper_style,
    SINGLE_COL_WIDTH,
    TWO_COL_WIDTH,
    PANEL_HEIGHT,
    TWO_PANEL_HEIGHT,
    get_model_palette,
)

apply_paper_style()   # call once before any plt.figure() / sns.* call
fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, PANEL_HEIGHT))
palette = get_model_palette(n_models)
"""

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Figure-size constants (all values in inches)
# ---------------------------------------------------------------------------
# Single-column article (e.g. JMLR, thesis, arXiv preprint) with ample space.
# US Letter / A4 with standard margins gives ~6.5 in text width.

SINGLE_COL_WIDTH: float = 4.0  # half-width figure in a single-column paper
TWO_COL_WIDTH: float = 6.5  # full text width in a single-column paper

PANEL_HEIGHT: float = 3.25  # standard single-metric panel
TWO_PANEL_HEIGHT: float = 3.5  # side-by-side two-subplot figure
TALL_PANEL_HEIGHT: float = 4.5  # taller panels (horizontal barplots with many labels)
GRID_CELL_SIZE: float = 2.5  # per-cell size for visualization grids


# ---------------------------------------------------------------------------
# Style application
# ---------------------------------------------------------------------------


def apply_paper_style(use_latex: bool = False) -> None:
    """Configure matplotlib + seaborn for publication-quality paper figures.

    Sets seaborn theme to ``style="ticks"`` with ``context="paper"``, then
    overrides rcParams for a single-column document with ample space:

    - 13 pt body font, 12 pt tick/legend labels
    - 1.5 pt line widths, 0.8 pt axis spines
    - No legend frame
    - PDF vector output (300 DPI raster fallback)

    By default enables ``text.usetex=True`` with Computer Modern roman font,
    which matches the body font of most NeurIPS/ICML LaTeX submissions. This
    requires a working TeX installation (e.g. ``texlive-full``).

    Args:
        use_latex: If True, render text via TeX with Computer
            Modern. Set to False to use DejaVu Sans — bundled with
            matplotlib, works on any machine with no TeX dependency.
    """
    # "ticks" removes the heavy white-grid background; font_scale=1.5 gives
    # 13 pt body text, comfortable in a wide single-column layout.
    sns.set_theme(style="ticks", context="paper", font_scale=1.5)

    rc: dict = {
        # --- Font (sans-serif fallback, overridden if use_latex=True) ---
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica Neue", "Helvetica", "Arial"],
        # --- Font sizes ---
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "axes.titlepad": 5.0,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        # --- Legend ---
        "legend.fontsize": 12,
        "legend.title_fontsize": 12,
        "legend.frameon": False,
        "legend.borderpad": 0.5,
        # --- Lines ---
        "lines.linewidth": 1.5,
        "lines.markersize": 6.0,
        # --- Axes geometry ---
        "axes.linewidth": 0.8,
        "axes.labelpad": 5.0,
        # --- Error bars ---
        "errorbar.capsize": 4.0,
        # --- Ticks ---
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # --- Figure / saving ---
        "figure.dpi": 150,  # screen preview; savefig uses savefig.dpi
        "savefig.dpi": 300,
        "savefig.format": "pdf",  # default vector output for LaTeX
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }

    if use_latex:
        rc.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
                "text.latex.preamble": r"\usepackage{lmodern}",
            }
        )

    plt.rcParams.update(rc)


# ---------------------------------------------------------------------------
# Color palette helper
# ---------------------------------------------------------------------------


def get_model_palette(n_models: int) -> list:
    """Return a colorblind-safe palette for ``n_models`` distinct series.

    Uses the Wong (2011) eight-color set via seaborn ``"colorblind"``,
    distinguishable under deuteranopia, protanopia, and tritanopia. Falls
    back to ``"tab10"`` when more than eight colors are needed.

    Args:
        n_models: Number of models / distinct series to color.

    Returns:
        List of RGB tuples of length ``n_models``.
    """
    if n_models <= 8:
        return sns.color_palette("colorblind", n_models)
    return sns.color_palette("tab10", n_models)
