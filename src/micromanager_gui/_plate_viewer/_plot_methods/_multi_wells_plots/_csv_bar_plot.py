from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

    from micromanager_gui._plate_viewer._graph_widgets import _MultilWellGraphWidget


CONDITION = "condition"
WEIGHTED_MEAN = "weighted_mean"
POOLED_SEM = "pooled_sem"
MEAN_SUFFIX = "_Mean"
SEM_SUFFIX = "_SEM"
N_SUFFIX = "_N"
EVK = "_evk_stim"
NON_EVK = "_evk_non_stim"
MEAN = "mean"
SEM = "sem"


class PlotData(TypedDict):
    """Type definition for plot data returned by parsing functions."""

    conditions: list[str]
    means: list[float]
    sems: list[float]
    fov_values_list: list[np.ndarray]
    pulse_length: str | None


def plot_csv_bar_plot(  # <- new name, call it however you like
    widget: _MultilWellGraphWidget,
    csv_path: str | Path,
    info: dict[str, str],
    mean_n_sem: bool = True,
) -> None:
    """Load a CSV file and create *bar* plots (mean ± pooled-SEM) per condition."""
    widget.figure.clear()
    if mean_n_sem:
        _create_bar_plot_mean_and_pooled_sem(widget, csv_path, info)
    else:
        _create_bar_plot(widget, csv_path, info)


def _create_bar_plot_mean_and_pooled_sem(
    widget: _MultilWellGraphWidget,
    csv_path: str | Path,
    info: dict[str, str],
) -> None:
    """Load a CSV file and create *bar* plots (mean ± pooled-SEM) per condition.

    The CSV must contain header triplets for each condition, named <Condition>_Mean,
    <Condition>_SEM, and <Condition>_N, with every row holding the corresponding values
    from a single FOV. For every condition the routine reconstructs these per-FOV
    statistics, computes a weighted mean (using N as weights) and a pooled SEM, then
    plots the mean +- SEM as bars, and overlays the individual FOV means as scatter
    points.
    """
    # parse data for triplet format
    data = _parse_csv_triplet_format(csv_path, info)
    if data is None:
        return

    # create the plot using shared plotting logic
    _create_shared_bar_plot(
        widget=widget,
        info=info,
        conditions=data["conditions"],
        means=data["means"],
        sems=data["sems"],
        fov_values_list=data["fov_values_list"],
        pulse_length=data["pulse_length"],
        bar_label="Weighted Mean ± Pooled SEM",
    )


def _create_bar_plot(
    widget: _MultilWellGraphWidget,
    csv_path: str | Path,
    info: dict[str, str],
) -> None:
    """
    Load a CSV and plot a bar plot with mean ± SEM per condition.

    The CSV should have a header with condition names, and each column
    should contain values for that condition. The function computes the mean
    and SEM for each condition, and plots them as bars with a scatter overlay
    of the raw FOV means.
    """
    # parse data for simple column format
    data = _parse_csv_column_format(csv_path, info)
    if data is None:
        return

    # create the plot using shared plotting logic
    _create_shared_bar_plot(
        widget=widget,
        info=info,
        conditions=data["conditions"],
        means=data["means"],
        sems=data["sems"],
        fov_values_list=data["fov_values_list"],
        pulse_length=None,
        bar_label="Mean ± SEM",
    )


def _parse_csv_triplet_format(
    csv_path: str | Path,
    info: dict[str, str],
) -> PlotData | None:
    """Parse CSV with triplet format (_Mean, _SEM, _N columns)."""
    parameter = info.get("parameter")
    if not parameter:
        return None

    evk = parameter in {"Stimulated Amplitude", "Non-Stimulated Amplitude"}
    pulse_length: str | None = None

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

    # get condition bases from _Mean columns
    mean_cols = [c for c in df.columns if c.endswith(MEAN_SUFFIX)]
    if not mean_cols:
        return None
    cond_bases = [c[:-5] for c in mean_cols]

    conditions = []
    means = []
    sems = []
    fov_values_list = []

    for base in cond_bases:
        col_mean, col_sem, col_n = (
            f"{base}{MEAN_SUFFIX}",
            f"{base}{SEM_SUFFIX}",
            f"{base}{N_SUFFIX}",
        )
        sub = df[[col_mean, col_sem, col_n]].dropna()
        if sub.empty:
            continue

        fov_means = sub[col_mean].to_numpy()
        fov_sems = sub[col_sem].to_numpy()
        Ns = sub[col_n].to_numpy()
        total_N = Ns.sum()

        if total_N <= 1:
            weighted_mean = fov_means.mean()
            pooled_sem = fov_sems.mean()
        else:
            weighted_mean = (fov_means * Ns).sum() / total_N
            within = ((Ns - 1) * fov_sems**2).sum()
            between = (Ns * (fov_means - weighted_mean) ** 2).sum()
            pooled_var = (within + between) / (total_N - 1)
            pooled_sem = np.sqrt(pooled_var) / np.sqrt(total_N)

        label = base
        # label cleaning for evoked traces
        if evk:
            label = label.replace(EVK, "").replace(NON_EVK, "")
            parts = label.split("_")
            pulse_length = parts[-1]  # "…_50"
            label = "_".join(parts[:-1])

        conditions.append(label)
        means.append(weighted_mean)
        sems.append(pooled_sem)
        fov_values_list.append(fov_means)

    if not conditions:
        return None

    return PlotData(
        conditions=conditions,
        means=means,
        sems=sems,
        fov_values_list=fov_values_list,
        pulse_length=pulse_length,
    )


def _parse_csv_column_format(
    csv_path: str | Path,
    info: dict[str, str],
) -> PlotData | None:
    """Parse CSV with simple column format (one column per condition)."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

    if df.empty:
        return None

    conditions = []
    means = []
    sems = []
    fov_values_list = []

    for col in df.columns:
        vals = df[col].dropna().to_numpy()
        if vals.size == 0:
            continue

        mean = float(vals.mean())
        sem = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0

        conditions.append(col)
        means.append(mean)
        sems.append(sem)
        fov_values_list.append(vals)

    if not conditions:
        return None

    return PlotData(
        conditions=conditions,
        means=means,
        sems=sems,
        fov_values_list=fov_values_list,
        pulse_length=None,
    )


def _create_shared_bar_plot(
    widget: _MultilWellGraphWidget,
    info: dict[str, str],
    conditions: list[str],
    means: list[float],
    sems: list[float],
    fov_values_list: list[np.ndarray],
    pulse_length: str | None,
    bar_label: str,
) -> None:
    """Shared plotting logic for both bar plot functions."""
    # Clear is handled by the main plot function
    ax: Axes = widget.figure.add_subplot(111)

    parameter = info.get("parameter", "")
    add_to_title = info.get("add_to_title", "")
    units = info.get("units", "")

    # handle condition toggles
    cond_list: dict[str, bool] = widget.conditions
    if not cond_list or len(cond_list) != len(conditions):
        cond_list = {cond: True for cond in conditions}
        widget.conditions = cond_list

    # filter based on toggles
    filtered_data = [
        (cond, mean, sem, fov_vals)
        for cond, mean, sem, fov_vals in zip(conditions, means, sems, fov_values_list)
        if cond_list.get(cond, True)
    ]

    if not filtered_data:
        return

    filtered_conditions, filtered_means, filtered_sems, filtered_fov_values = map(
        list, zip(*filtered_data)
    )

    # create bar plot
    x = np.arange(1, len(filtered_conditions) + 1)
    ax.bar(
        x,
        filtered_means,
        yerr=filtered_sems,
        capsize=5,
        zorder=2,
        color="lightgray",
        edgecolor="black",
    )

    # scatter overlay of individual FOV values
    for idx, vals in enumerate(filtered_fov_values):
        jitter = np.random.normal(0, 0.05, size=vals.size)
        ax.scatter(
            np.full(vals.size, x[idx]) + jitter,
            vals,
            marker="o",
            color="black",
            edgecolors="black",
            zorder=3,
        )

    # set labels and formatting
    ax.set_xticks(x)
    ax.set_xticklabels(filtered_conditions, rotation=45, ha="right")

    units_text = f" ({units})" if units else ""
    ax.set_ylabel(f"{parameter}{units_text}")

    # handle title with pulse length
    title = f"{parameter} per Condition {add_to_title}"
    if pulse_length is not None:
        title = title.rstrip(")")
        title += f" - {pulse_length} ms Pulse)"
    ax.set_title(title)

    ax.grid(True, alpha=0.3)

    # create legend
    bar_handle = Rectangle(
        (0, 0),
        1,
        1,
        facecolor="lightgray",
        edgecolor="black",
        label=bar_label,
    )
    scatter_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="black",
        markeredgecolor="black",
        markersize=6,
        label="Individual FOV",
    )
    ax.legend(handles=[bar_handle, scatter_handle], loc="upper left", frameon=True)

    widget.figure.tight_layout()
    widget.canvas.draw()
