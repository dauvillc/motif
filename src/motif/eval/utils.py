from typing import Any

import numpy as np
from pandas import Timedelta


def format_tdelta(tdelta: Timedelta) -> str:
    """Formats a timedelta object to HH:MM format"""
    total_seconds = int(tdelta.total_seconds())
    sign = "-" if total_seconds < 0 else ""
    total_seconds = abs(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    if hours == 0:
        return f"{sign}{minutes} min"
    return f"{sign}{hours:02d}h {minutes:02d} min"


def set_log_ylim_from_artists(ax: Any, margin: float = 0.1) -> None:
    """Set y-axis limits from the rendered artists (mean lines and CI bands),
    with a margin in log space.

    Seaborn's default autoscaling uses the full raw data range rather than the
    range of the plotted aggregated statistics (mean ± CI), which can make the
    scale far too wide. This method reads the y-data directly from the rendered
    Line2D and PathCollection objects and sets tighter limits.

    Args:
        ax: The matplotlib axes to adjust.
        margin: Fractional margin to add on each side of the data range in log space.
    """
    y_vals = []
    for line in ax.get_lines():
        y_vals.extend(line.get_ydata())  # type: ignore
    for coll in ax.collections:
        for path in coll.get_paths():
            y_vals.extend(path.vertices[:, 1])  # type: ignore
    y_vals = np.array([v for v in y_vals if np.isfinite(v) and v > 0])
    if len(y_vals) == 0:
        return
    log_min = np.log10(y_vals.min())
    log_max = np.log10(y_vals.max())
    log_range = log_max - log_min
    ax.set_ylim(
        10 ** (log_min - margin * log_range),
        10 ** (log_max + margin * log_range),
    )
