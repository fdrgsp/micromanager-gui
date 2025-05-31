from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

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
