"""
Implements a footprint coverage visualization across all evaluation samples.

Produces a single cartopy map figure showing the geographic extent of every
source observation (input and target) across all samples.  Each unique source
name is assigned a distinct color.  Because satellite swaths are not
rectangular, the spatial footprint of each swath is computed as the convex
hull of its (lat, lon) point cloud and drawn as a semi-transparent polygon.

Usage in Hydra config::

    evaluation_classes:
      footprint_map:
        _target_: motif.eval.footprint_visualization.FootprintVisualizationEval
        eval_fraction: 1.0
        point_stride: 5
        alpha: 0.15
        sample_index: null  # set to an integer to visualize a single sample
"""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import ConvexHull, QhullError
from tqdm import tqdm

from motif.eval.abstract_evaluation_metric import AbstractMultisourceEvaluationMetric
from motif.eval.plot_style import TWO_COL_WIDTH, apply_paper_style, get_model_palette


class FootprintVisualizationEval(AbstractMultisourceEvaluationMetric):
    """Evaluation class that produces a single cartopy map of source footprints.

    Iterates all (or a random subset of) samples, collects the lat/lon extents
    of every source observation, and plots them as convex-hull polygons over a
    global cartopy PlateCarree map.  Input sources and target sources are
    distinguished by line style; each unique source name gets its own color.
    """

    def __init__(
        self,
        model_data: dict[str, dict],
        parent_results_dir: str | Path,
        eval_fraction: float = 1.0,
        point_stride: int = 5,
        alpha: float = 0.15,
        sample_index: int | None = None,
        **kwargs,
    ):
        """
        Args:
            model_data: Dictionary mapping model_ids to model specifications.
            parent_results_dir: Parent directory for all results.
            eval_fraction: Fraction of samples to include (0.0 to 1.0).
                Ignored when ``sample_index`` is set.
            point_stride: Spatial subsampling stride applied to the lat/lon
                grids before convex-hull computation.  Higher values are faster
                but less precise for irregular swath edges.
            alpha: Fill transparency for the hull polygons (0 = fully
                transparent, 1 = fully opaque).
            sample_index: If given, only the sample at this 0-based iteration
                index is processed.  Takes precedence over ``eval_fraction``.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """
        super().__init__(
            id_name="footprint_map",
            full_name="Footprint Map",
            model_data=model_data,
            parent_results_dir=parent_results_dir,
            **kwargs,
        )
        self.eval_fraction = eval_fraction
        self.point_stride = point_stride
        self.alpha = alpha
        self.sample_index = sample_index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, **kwargs):
        """Collect footprints from all samples and render a single map figure."""
        # footprints[source_name] = list of (lat_1d, lon_1d) tuples
        footprints: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        # Track whether each source name is used as input, target, or both.
        # Maps source_name -> set of role strings ("input" | "target")
        roles: dict[str, set[str]] = {}

        if self.sample_index is not None:
            n_samples = 1
            selected = {self.sample_index}
            iterator = (item for i, item in enumerate(self.samples_iterator()) if i in selected)
        elif self.eval_fraction < 1.0:
            n_samples = int(self.eval_fraction * self.n_samples)
            rng = np.random.default_rng(seed=42)
            selected = set(rng.choice(self.n_samples, size=n_samples, replace=False))
            iterator = (item for i, item in enumerate(self.samples_iterator()) if i in selected)
        else:
            n_samples = self.n_samples
            iterator = self.samples_iterator()

        for sample_df, true_obs, _preds in tqdm(
            iterator, desc="Collecting footprints", total=n_samples
        ):
            # Use the first model's observations (footprints are identical across models).
            any_model = next(iter(true_obs.keys()))
            for src, dataset in true_obs[any_model].items():
                avail = sample_df.loc[(any_model, src.name, src.index), "avail"]
                if avail == -1:
                    continue
                role = "target" if avail == 0 else "input"

                lat_2d = dataset["lat"].values  # (H, W)
                lon_2d = dataset["lon"].values  # (H, W)

                # Subsample for performance, then flatten.
                s = self.point_stride
                lat_flat = lat_2d[::s, ::s].ravel()
                lon_flat = lon_2d[::s, ::s].ravel()

                # Remove NaN coordinates.
                valid = ~(np.isnan(lat_flat) | np.isnan(lon_flat))
                lat_flat = lat_flat[valid]
                lon_flat = lon_flat[valid]

                if lat_flat.size < 3:
                    continue  # too few points for a hull

                footprints.setdefault(src.name, []).append((lat_flat, lon_flat))
                roles.setdefault(src.name, set()).add(role)

        if not footprints:
            return  # nothing to plot

        self._make_figure(footprints, roles)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_figure(
        self,
        footprints: dict[str, list[tuple[np.ndarray, np.ndarray]]],
        roles: dict[str, set[str]],
    ):
        apply_paper_style()

        source_names = sorted(footprints.keys())
        colors = get_model_palette(len(source_names))
        color_map = dict(zip(source_names, colors))

        fig, ax = plt.subplots(
            1,
            1,
            figsize=(TWO_COL_WIDTH, TWO_COL_WIDTH * 0.55),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        extent = self._compute_extent(footprints)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5, zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="aliceblue", alpha=0.5, zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=1)
        ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle="--")

        legend_handles: dict[str, plt.Artist] = {}  # type: ignore[type-arg]

        for src_name in source_names:
            color = color_map[src_name]
            src_roles = roles.get(src_name, set())
            # Dashed edge for target-only sources, solid for inputs.
            linestyle = "--" if src_roles == {"target"} else "-"
            label = self._display_src_name(src_name)

            for idx, (lat_flat, lon_flat) in enumerate(footprints[src_name]):
                hull_polygon = self._convex_hull_polygon(lat_flat, lon_flat)
                if hull_polygon is None:
                    continue

                patch = MplPolygon(
                    hull_polygon,
                    closed=True,
                    facecolor=(*color[:3], self.alpha),
                    edgecolor=(*color[:3], min(self.alpha * 4, 0.9)),
                    linewidth=0.4,
                    linestyle=linestyle,
                    transform=ccrs.PlateCarree(),
                    zorder=2,
                )
                ax.add_patch(patch)

                if src_name not in legend_handles:
                    legend_handles[src_name] = MplPolygon(
                        [[0, 0]],
                        closed=True,
                        facecolor=(*color[:3], max(self.alpha * 3, 0.4)),
                        edgecolor=color[:3],
                        linewidth=0.8,
                        linestyle=linestyle,
                        label=label,
                    )

        if legend_handles:
            ax.legend(
                handles=list(legend_handles.values()),
                loc="lower left",
                fontsize=7,
                frameon=True,
                framealpha=0.8,
            )

        ax.set_title("Source Footprint Coverage", fontsize=10)

        stem = self.metric_results_dir / "footprint_map"
        fig.savefig(str(stem) + ".pdf")
        fig.savefig(str(stem) + ".svg", format="svg")
        plt.close(fig)

    @staticmethod
    def _compute_extent(
        footprints: dict[str, list[tuple[np.ndarray, np.ndarray]]],
        margin: float = 10.0,
    ) -> list[float]:
        all_lats = np.concatenate([lat for pts in footprints.values() for lat, _ in pts])
        all_lons = np.concatenate([lon for pts in footprints.values() for _, lon in pts])
        lat_min = max(float(all_lats.min()) - margin, -90.0)
        lat_max = min(float(all_lats.max()) + margin, 90.0)
        lon_min = max(float(all_lons.min()) - margin, -180.0)
        lon_max = min(float(all_lons.max()) + margin, 180.0)
        return [lon_min, lon_max, lat_min, lat_max]

    @staticmethod
    def _convex_hull_polygon(lat_flat: np.ndarray, lon_flat: np.ndarray) -> np.ndarray | None:
        """Return the closed convex hull as an (N+1, 2) array of (lon, lat) pairs.

        Returns None when the hull cannot be computed (too few points, degenerate
        geometry, or antimeridian crossing that would produce an unreasonably wide
        polygon).
        """
        pts = np.column_stack([lon_flat, lat_flat])  # (lon, lat) for cartopy
        try:
            hull = ConvexHull(pts)
        except QhullError:
            return None

        hull_pts = pts[hull.vertices]
        # Close the polygon.
        hull_pts = np.vstack([hull_pts, hull_pts[0]])

        # Sanity-check: reject polygons that span > 180° in longitude (antimeridian).
        lon_span = hull_pts[:, 0].max() - hull_pts[:, 0].min()
        if lon_span > 180:
            return None

        return hull_pts
