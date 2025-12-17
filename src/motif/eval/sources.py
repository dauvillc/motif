"""Implements the SourcesRepresentation class."""

from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from motif.eval.abstract_evaluation_metric import AbstractMultisourceEvaluationMetric


class SourcesRepresentationEvaluation(AbstractMultisourceEvaluationMetric):
    """Evaluation class that analyzes the evaluation samples dataframe
    to determine:
    * which sources are available in each sample
    * which source is masked
    The class then displays that information through various plots.
    """

    def __init__(
        self,
        model_data,
        parent_results_dir,
        num_workers=1,
        include_quantitative_results=True,
        **kwargs,
    ):
        super().__init__(
            id_name="sources_representation",
            full_name="Sources Representation Evaluation",
            model_data=model_data,
            parent_results_dir=parent_results_dir,
            **kwargs,
        )
        self.num_workers = num_workers
        self.quant_results = None
        if include_quantitative_results:
            # Check that the results file exists
            results_file = self.parent_results_dir / "quantitative" / "full_results.json"
            if results_file.exists():
                self.quant_results = pd.read_json(
                    results_file,
                    orient="records",
                    lines=True,
                )
            else:
                print(
                    f"Quantitative results file {results_file} not found. "
                    "Skipping quantitative results inclusion."
                )

    def evaluate(self, **kwargs):
        """Main evaluation method. Uses self.samples_df, a DataFrame created in the
        parent class's init(), and containing the following columns:
        * sample_index: index of the sample in the dataset
        * source_name, source_index: tuple of (str, int) identifying one source
            in the sample, and its order of appearance. For example, (S, 0)
            means this is the first and latest obs from source S in the sample,
            (S, 3) means this is the fourth latest obs from source S in the sample.
        * avail (int): binary value indicating whether this source is available (1)
            or masked (0) in the sample.
        * dt (float): time difference (in hours) between the source observation
            time and the reference time of the sample.
        """

        # Configure plotting style for publication quality
        sns.set_theme(style="whitegrid", context="poster")  # Use poster context for larger fonts
        plt.rcParams.update(
            {
                "font.size": 14,
                "axes.labelsize": 16,
                "axes.titlesize": 18,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
                "legend.fontsize": 14,
                "figure.dpi": 300,  # High resolution
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "DejaVu Sans"],
            }
        )
        self._compute_combinations_statistics()

    def _compute_combinations_statistics(self):
        """Compute statistics about the source combinations in the evaluation set,
        and create plots to visualize them.
        """
        # For now, we'll ignore multiple appearances of the same source in a sample.
        samples_df = self.samples_df.drop_duplicates(subset=["sample_index", "source_name"])
        grouped_df = samples_df.groupby("sample_index")

        # 1. We want to know which unique combinations of sources exist, and
        # how many samples correspond to each combination.
        # - obtain a series of one row per sample index, with each value being the list
        # of sources present in the sample sorted alphabetically.
        sample_combinations = grouped_df.apply(lambda grp: sorted(list(grp["source_name"])))
        # - for each row that has 3 or more sources, convert to one row per combination of 2
        sample_combinations = sample_combinations.apply(
            lambda src_list: (
                list(combinations(src_list, 2)) if len(src_list) >= 3 else [tuple(src_list)]
            )
        ).explode()
        # - count the occurrences of each unique combination
        combination_counts = sample_combinations.value_counts()
        # - sort by decreasing count
        combination_counts = combination_counts.sort_values(ascending=False)
        # - compute the fraction of samples for each combination
        combination_counts = combination_counts.to_frame(name="count")
        combination_counts["fraction"] = (
            combination_counts["count"] / combination_counts["count"].sum()
        )
        # convert the source names to their display names for better readability
        combination_counts.index = combination_counts.index.map(
            lambda src_list: ", ".join(self._display_src_name(s) for s in src_list)
        )
        # We'll make two horizontal bar plots, one for the counts and one for the fractions.
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        sns.barplot(
            x="count",
            y=combination_counts.index.astype(str),
            data=combination_counts.reset_index(),
            ax=ax,
        )
        ax.set_title("Counts of source combinations in evaluation samples")
        ax.set_xlabel("Count")
        ax.set_ylabel("Source combination")
        # Reduce the font size of the y-axis labels as the combinations can be long
        ax.tick_params(axis="y", labelsize=8)
        plt.tight_layout()
        fig.savefig(self.metric_results_dir / "source_combinations_counts.png")
        plt.close(fig)

        if self.quant_results is not None:
            # If using quantitative results, we'll make a two-subplot figure
            # with on the right the counts and on the left the results of each model
            # for each combination.
            quant = (
                self.quant_results.drop(["source_name", "source_index", "channel"], axis="columns")
                .groupby(["model_id", "sample_index"])
                .mean()
                .reset_index()
            )
            quant = quant.merge(
                sample_combinations.to_frame(name="combination"),
                on="sample_index",
            )
            quant["combination"] = quant["combination"].map(
                lambda src_list: ", ".join(self._display_src_name(s) for s in src_list)
            )
            quant = (
                quant.drop("sample_index", axis="columns")
                .groupby(["model_id", "combination"])
                .mean()
            ).reset_index()
            # Re-order the combinations so that they match the order in combination_counts
            quant = quant.set_index("combination").loc[combination_counts.index].reset_index()
            quant = quant.rename(columns={"index": "combination"})

            fig, ax = plt.subplots(1, 2, figsize=(10, 8))
            sns.barplot(
                x="crps",
                y="combination",
                hue="model_id",
                data=quant,
                ax=ax[0],
            )
            ax[0].set_title("CRPS by source combination")
            ax[0].set_xlabel("CRPS")
            ax[0].set_ylabel("Source combination")
            ax[0].tick_params(axis="y", labelsize=8)
            # Disabled the axis's legend as we'll have a global one
            ax[0].legend_.remove()
            sns.barplot(
                x="count",
                y=combination_counts.index.astype(str),
                data=combination_counts.reset_index(),
                ax=ax[1],
            )
            ax[1].set_title("Counts of source combinations")
            ax[1].set_xlabel("Count")
            ax[1].set_ylabel("Source combination")
            ax[1].tick_params(axis="y", labelsize=8)
            # Make sure the legend does not overlap with the plots
            fig.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 0.95),
                ncol=len(quant["model_id"].unique()),
            )
            plt.tight_layout()
            fig.savefig(self.metric_results_dir / "source_combinations_crps.png")
            plt.close(fig)

        return combination_counts
