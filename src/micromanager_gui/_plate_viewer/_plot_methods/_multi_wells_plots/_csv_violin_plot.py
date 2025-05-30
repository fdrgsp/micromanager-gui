from __future__ import annotations

from typing import TYPE_CHECKING

import mplcursors
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

    from micromanager_gui._plate_viewer._graph_widgets import _MultilWellGraphWidget


def plot_csv_violin_plot(
    widget: _MultilWellGraphWidget, csv_path: str | Path = ""
) -> None:
    """Load a CSV file and create violin plots with conditions on the x-axis.

    Parameters
    ----------
    widget : _MultilWellGraphWidget
        The widget to plot on.
    csv_path : str | Path | None
        Path to the CSV file. If None, opens a file dialog.
    """
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    if not csv_path:
        return

    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)

        # Extract all condition columns (those ending with "_Mean")
        mean_columns = [col for col in df.columns if col.endswith("_Mean")]

        if not mean_columns:
            ax.text(
                0.5,
                0.5,
                'No "_Mean" columns found in CSV',
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            widget.canvas.draw()
            return

        # Prepare data for violin plot
        plot_data = []
        condition_labels = []

        for col in mean_columns:
            # Get the data for this condition (remove NaN values)
            values = df[col].dropna().values
            if len(values) > 0:
                plot_data.append(values)
                # Clean up condition name (remove "_Mean" suffix)
                condition_name = col.replace("_Mean", "")
                condition_labels.append(condition_name)

        if not plot_data:
            ax.text(
                0.5,
                0.5,
                "No valid data found in CSV",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            widget.canvas.draw()
            return

        # Create violin plot
        violin_parts = ax.violinplot(
            plot_data,
            positions=range(1, len(plot_data) + 1),
            showmeans=True,
            showmedians=True,
        )

        # Customize appearance
        for pc in violin_parts["bodies"]:
            pc.set_facecolor("lightblue")
            pc.set_alpha(0.7)

        # Set labels and title
        ax.set_xticks(range(1, len(condition_labels) + 1))
        ax.set_xticklabels(condition_labels, rotation=45, ha="right")
        ax.set_xlabel("Conditions")
        ax.set_ylabel("Amplitude")
        ax.set_title("Violin Plot of Conditions")

        # Add grid for better readability
        ax.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = []
        for data, label in zip(plot_data, condition_labels):
            n = len(data)
            mean = np.mean(data)
            std = np.std(data)
            stats_text.append(f"{label}: n={n}, μ={mean:.3f}, std={std:.3f}")

        # Add statistics as text box
        stats_str = "\n".join(stats_text)
        ax.text(
            0.02,
            0.98,
            stats_str,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    except Exception as e:
        ax.text(
            0.5,
            0.5,
            f"Error loading CSV: {e!s}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    widget.figure.tight_layout()

    # Add hover functionality to display condition info
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        # Get the x position to determine which condition
        x_pos = sel.target[0]
        condition_idx = round(x_pos) - 1

        if 0 <= condition_idx < len(condition_labels):
            condition = condition_labels[condition_idx]
            data = plot_data[condition_idx]
            y_val = sel.target[1]

            # Find closest data point
            closest_idx = np.argmin(np.abs(data - y_val))
            actual_val = data[closest_idx]

            sel.annotation.set(
                text=f"{condition}\nValue: {actual_val:.3f}\nn={len(data)}",
                fontsize=8,
                color="black",
            )

    widget.canvas.draw()


def load_and_plot_csv_violin(widget: _MultilWellGraphWidget) -> None:
    """Load and plot CSV violin plot with file dialog.

    Parameters
    ----------
    widget : _MultilWellGraphWidget
        The widget to plot on.
    """
    plot_csv_violin_plot(widget)
