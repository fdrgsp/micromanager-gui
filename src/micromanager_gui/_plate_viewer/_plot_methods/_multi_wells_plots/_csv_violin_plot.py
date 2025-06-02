from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

    from micromanager_gui._plate_viewer._graph_widgets import _MultilWellGraphWidget


def plot_csv_violin_plot(
    widget: _MultilWellGraphWidget,
    csv_path: str | Path,
    info: dict[str, str],
    mean_n_sem: bool = True,
) -> None:
    """Load a CSV file and create violin plots with conditions on the x-axis.

    Parameters
    ----------
    widget : _MultilWellGraphWidget
        The widget to plot on.
    csv_path : str | Path | None
        Path to the CSV file.
    info : dict[str, str]
        Additional information, not used in this function.
    mean_n_sem : bool, optional
        If True, means that the CSV file contains for each condition 3 columns,
        the mean, the number of samples, and the standard error of the mean (SEM).
        By default True.
    """
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    parameter = info.get("parameter")
    if not parameter:
        return

    cond_list: dict[str, bool] = widget.conditions

    add_to_title = info.get("add_to_title", "")
    units = info.get("units", "")

    evk = parameter in {"Stimulated Amplitude", "Non-Stimulated Amplitude"}
    pulse_length: str | None = None

    # load the CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    if mean_n_sem:
        # extract all condition columns (those ending with "_Mean")
        mean_columns = [col for col in df.columns if col.endswith("_Mean")]

        if not mean_columns:
            return

        # Prepare data for violin plot
        plot_data = []
        condition_labels = []

        for col in mean_columns:
            # get the data for this condition (remove NaN values)
            values = df[col].dropna().values
            if len(values) > 0:
                plot_data.append(values)
                # clean up condition name
                condition_name = col.replace("_Mean", "")
                if evk:
                    condition_name = condition_name.replace("_evk_stim", "")
                    condition_name = condition_name.replace("_evk_non_stim", "")
                    condition_name_split = condition_name.split("_")
                    pulse_length = str(condition_name_split[-1])
                    condition_name = "_".join(condition_name_split[:-1])
                condition_labels.append(condition_name)

    else:
        plot_data = [df[col].dropna().values for col in df.columns]
        condition_labels = list(df.columns)

    if not cond_list or len(cond_list) != len(condition_labels):
        cond_list = {label: True for label in condition_labels}
        widget.conditions = cond_list

    # filter plot_data and condition_labels to only include selected conditions
    plot_data = [
        data
        for label, data in zip(condition_labels, plot_data)
        if label in cond_list and cond_list[label] is True
    ]
    condition_labels = [
        label
        for label in condition_labels
        if label in cond_list and cond_list[label] is True
    ]
    if not plot_data:
        return

    # create violin plot
    parts = ax.violinplot(
        plot_data,
        positions=range(1, len(plot_data) + 1),
        showmeans=True,
        showmedians=True,
        showextrema=False,
        side="both",
    )

    for body in parts.get("bodies", []):
        body.set_facecolor("lightgray")
        body.set_edgecolor("black")
        body.set_alpha(0.5)

    lines = {"cmeans": "green", "cmedians": "magenta"}
    for key, color in lines.items():
        parts[key].set_color(color)
        parts[key].set_linewidth(1.3)

    # add mean and median legend
    ax.plot([], [], color="green", label="Mean", linewidth=1.3)
    ax.plot([], [], color="magenta", label="Median", linewidth=1.3)
    ax.legend(loc="upper left", frameon=True)

    # labels and title
    ax.set_xticks(range(1, len(condition_labels) + 1))
    ax.set_xticklabels(condition_labels, rotation=45, ha="right")
    ax.set_xlabel("Conditions")
    units = f"({units})" if units else ""
    ax.set_ylabel(f"{parameter} {units}")
    if pulse_length is not None:
        add_to_title = add_to_title[:-1]
        add_to_title += f" - {pulse_length} ms Pulse)"
    ax.set_title(f"{parameter} per Conditions {add_to_title}")

    ax.grid(True, alpha=0.3)

    widget.figure.tight_layout()

    widget.canvas.draw()


def plot_csv_bar_plot(  # <- new name, call it however you like
    widget: _MultilWellGraphWidget,
    csv_path: str | Path,
    info: dict[str, str],
    mean_n_sem: bool = True,
) -> None:
    """
    Load a CSV file and create *bar* plots (mean ± pooled-SEM) per condition.

    Parameters
    ----------
    widget, csv_path, info, mean_n_sem
        Same meaning as in the original function.
    """
    widget.figure.clear()

    if not mean_n_sem:
        return

    ax: Axes = widget.figure.add_subplot(111)

    parameter = info.get("parameter")
    if not parameter:
        return

    cond_list: dict[str, bool] = widget.conditions  # saved toggles

    add_to_title = info.get("add_to_title", "")
    units = info.get("units", "")

    evk = parameter in {"Stimulated Amplitude", "Non-Stimulated Amplitude"}
    pulse_length: str | None = None

    # ------------------------------------------------------------------ #
    # 1) Read CSV
    # ------------------------------------------------------------------ #
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    # ------------------------------------------------------------------ #
    # 2) Detect condition bases
    # ------------------------------------------------------------------ #
    mean_cols = [c for c in df.columns if c.endswith("_Mean")]
    if not mean_cols:
        return
    cond_bases = [c[:-5] for c in mean_cols]  # strip "_Mean"

    # Lists for plotting
    summary = []  # each item: dict(condition, weighted_mean, pooled_sem)
    fov_means_list = []  # each item: np.ndarray of per-FOV means (for scatter)
    clean_labels = []

    # ------------------------------------------------------------------ #
    # 3) Aggregate stats per condition
    # ------------------------------------------------------------------ #
    for base in cond_bases:
        col_mean, col_sem, col_n = f"{base}_Mean", f"{base}_SEM", f"{base}_N"
        sub = df[[col_mean, col_sem, col_n]].dropna()
        if sub.empty:
            continue

        means = sub[col_mean].to_numpy()
        sems = sub[col_sem].to_numpy()
        Ns = sub[col_n].to_numpy()
        total_N = Ns.sum()

        if total_N <= 1:
            weighted_mean = means.mean()
            pooled_sem = sems.mean()
        else:
            weighted_mean = (means * Ns).sum() / total_N
            within = ((Ns - 1) * sems**2).sum()
            between = (Ns * (means - weighted_mean) ** 2).sum()
            pooled_var = (within + between) / (total_N - 1)
            pooled_sem = np.sqrt(pooled_var) / np.sqrt(total_N)

        # Label cleaning for evoked traces
        label = base
        if evk:
            label = label.replace("_evk_stim", "").replace("_evk_non_stim", "")
            parts = label.split("_")
            pulse_length = parts[-1]  # “…_50”
            label = "_".join(parts[:-1])

        summary.append(
            {
                "condition": label,
                "weighted_mean": weighted_mean,
                "pooled_sem": pooled_sem,
            }
        )
        fov_means_list.append(means)  # keep raw means for scatter
        clean_labels.append(label)

    if not summary:
        return

    # ------------------------------------------------------------------ #
    # 4) Condition toggles from the widget
    # ------------------------------------------------------------------ #
    if not cond_list or len(cond_list) != len(clean_labels):
        cond_list = {lab: True for lab in clean_labels}
        widget.conditions = cond_list

    # Respect toggles
    filtered = [
        (s, fov)
        for s, fov in zip(summary, fov_means_list)
        if cond_list.get(s["condition"], True)
    ]
    if not filtered:
        return

    summary, fov_means_list = zip(*filtered)  # unzip

    conditions = [s["condition"] for s in summary]
    means = [s["weighted_mean"] for s in summary]
    sems = [s["pooled_sem"] for s in summary]

    # ------------------------------------------------------------------ #
    # 5) Plot bars
    # ------------------------------------------------------------------ #
    x = np.arange(1, len(conditions) + 1)
    ax.bar(x, means, yerr=sems, capsize=6, zorder=2)

    # ------------------------------------------------------------------ #
    # 6) Scatter overlay of individual FOV means
    #     (small horizontal jitter so points don't stack)
    # ------------------------------------------------------------------ #
    for idx, means_fov in enumerate(fov_means_list):
        jitter = np.random.normal(0, 0.05, size=means_fov.size)
        ax.scatter(
            np.full(means_fov.size, x[idx]) + jitter,
            means_fov,
            alpha=0.6,
            marker="o",
            zorder=3,
        )

    # ------------------------------------------------------------------ #
    # 7) Cosmetics
    # ------------------------------------------------------------------ #
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha="right")

    units_text = f" ({units})" if units else ""
    ax.set_ylabel(f"{parameter}{units_text}")

    if pulse_length is not None:
        add_to_title = add_to_title.rstrip(")")
        add_to_title += f" – {pulse_length} ms Pulse)"

    ax.set_title(f"{parameter} per Condition {add_to_title}")
    ax.grid(True, alpha=0.3)

    # Legend: bar + SEM proxy, scatter proxy
    eb_proxy = ax.errorbar([], [], yerr=[], fmt=" ", capsize=6, label="pooled SEM")[0]
    sc_proxy = ax.scatter([], [], marker="o", alpha=0.6, label="individual FOV")
    ax.legend(handles=[eb_proxy, sc_proxy], loc="upper left", frameon=True)

    widget.figure.tight_layout()
    widget.canvas.draw()
