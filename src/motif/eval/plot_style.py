"""Shared publication-quality plotting style for NeurIPS / ICML / ECML figures.

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
# NeurIPS two-column text width: 5.5 in; single column: 3.25 in
# ICML single-column text width: ~6.75 in
# For safety, TWO_COL_WIDTH = 5.5 covers both venues as a full-width figure.

SINGLE_COL_WIDTH: float = 3.25  # half-width in a two-column paper
TWO_COL_WIDTH: float = 5.5  # full text width in a two-column paper

PANEL_HEIGHT: float = 2.25  # standard single-metric panel
TWO_PANEL_HEIGHT: float = 2.5  # side-by-side two-subplot figure
TALL_PANEL_HEIGHT: float = 3.25  # taller panels (horizontal barplots with many labels)
GRID_CELL_SIZE: float = 1.75  # per-cell size for visualization grids (was 3.0)


# ---------------------------------------------------------------------------
# Style application
# ---------------------------------------------------------------------------


def apply_paper_style(use_latex: bool = False) -> None:
    """Configure matplotlib + seaborn for publication-quality paper figures.

    Sets seaborn theme to ``style="ticks"`` with ``context="paper"``, then
    overrides rcParams for A4 single-column document conventions:

    - 11 pt body font, 10 pt tick/legend labels
    - 1 pt line widths, 0.6 pt axis spines
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
    # "ticks" removes the heavy white-grid background; font_scale=1.3 gives
    # 11 pt body text, readable in A4 single-column documents.
    sns.set_theme(style="ticks", context="paper", font_scale=1.3)

    rc: dict = {
        # --- Font (sans-serif fallback, overridden if use_latex=True) ---
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica Neue", "Helvetica", "Arial"],
        # --- Font sizes ---
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "axes.titlepad": 4.0,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        # --- Legend ---
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
        "legend.frameon": False,
        "legend.borderpad": 0.4,
        # --- Lines ---
        "lines.linewidth": 1.0,
        "lines.markersize": 4.0,
        # --- Axes geometry ---
        "axes.linewidth": 0.6,
        "axes.labelpad": 4.0,
        # --- Error bars ---
        "errorbar.capsize": 3.0,
        # --- Ticks ---
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
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
