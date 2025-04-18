from __future__ import annotations

from typing import TYPE_CHECKING

import mplcursors

# from micromanager_gui._plate_viewer._util import get_connectivity

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from micromanager_gui._plate_viewer._graph_widgets import _MultilWellGraphWidget
    from micromanager_gui._plate_viewer._util import ROIData


COUNT_INCREMENT = 1
DEFAULT_COLOR = "gray"
STIMULATED_COLOR = "green"
NON_STIMULATED_COLOR = "magenta"


def _plot_multi_well_data(
    widget: _MultilWellGraphWidget,
    data: dict[str, dict[str, ROIData]],
    # positions: list[int] | None = None,
    amp: bool = False,
    freq: bool = False,
    iei: bool = False,
    cell_size: bool = False,
    sync: bool = False,
) -> None:
    """Plot the multi-well data."""
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    well_count: int = 0
    wells_names: list[str] = []
    # loop over the data and plot the data
    for well_and_fov, roi_data in data.items():
        # this is to store the well names to then use them as x axis labels
        if (well := well_and_fov.split("_")[0]) not in wells_names:
            wells_names.append(well)
            # we increment the well count only when we find a new well
            well_count += 1

        for roi in roi_data:
            r_data = roi_data[roi]

            if amp and freq:
                if (
                    r_data.peaks_amplitudes_dec_dff is None
                    or r_data.dec_dff_frequency is None
                ):
                    continue
                amp_list = r_data.peaks_amplitudes_dec_dff
                well_freq_list = [r_data.dec_dff_frequency] * len(amp_list)
                ax.plot(
                    amp_list, well_freq_list, "o", label=f"{well_and_fov} - roi{roi}"
                )
                ax.set_xlabel("Amplitude")
                ax.set_ylabel("Frequency")

            elif amp:
                if r_data.peaks_amplitudes_dec_dff is None:
                    continue
                ax.plot(
                    [well_count] * len(r_data.peaks_amplitudes_dec_dff),
                    r_data.peaks_amplitudes_dec_dff,
                    "o",
                    label=f"{well_and_fov} - roi{roi}",
                )
                ax.set_xlabel("Wells")
                ax.set_ylabel("Amplitude")

            elif freq:
                if r_data.dec_dff_frequency is None:
                    continue
                ax.plot(
                    well_count,
                    r_data.dec_dff_frequency,
                    "o",
                    label=f"{well_and_fov} - roi{roi}",
                )
                ax.set_xlabel("Wells")
                ax.set_ylabel("Frequency")

            elif iei:
                if r_data.iei is None:
                    continue
                ax.plot(
                    [well_count] * len(r_data.iei),
                    r_data.iei,
                    "o",
                    label=f"{well_and_fov} - roi{roi}",
                )
                ax.set_xlabel("Wells")
                ax.set_ylabel("Inter-event intervals (sec)")

            elif cell_size:
                if r_data.cell_size is None:
                    continue
                ax.plot(
                    well_count,
                    r_data.cell_size,
                    "o",
                    label=f"{well_and_fov} - roi{roi}",
                )
                ax.set_xlabel("Wells")
                ax.set_ylabel(f"Cell Size ({r_data.cell_size_units})")

    # set ticks labels as well names
    if (amp or freq or iei or cell_size) and not (amp and freq):
        ax.set_xticks(range(1, len(wells_names) + 1))
        ax.set_xticklabels(wells_names)

    widget.figure.tight_layout()

    # add hover functionality to display the well position
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        sel.annotation.set(text=sel.artist.get_label(), fontsize=8, color="black")

    widget.canvas.draw()


def _plot_multi_cond_data(
    widget: _MultilWellGraphWidget,
    data: dict[str, list[ROIData]],
    cond_ordered: list[str],
    plate_map_color: dict[str, str],
    amp: bool = False,
    freq: bool = False,
    iei: bool = False,
    cell_size: bool = False,
    sync: bool = False,
) -> None:
    """Make plots based on conditions selected."""
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    dp_dict: dict[str, list[float]] = {}

    for cond, roi_data in data.items():
        # well = well_and_fov.split("_")[0]
        # if cond_sel is not None and well not in cond_sel:
        #     continue

        if cond not in dp_dict:
            dp_dict[cond] = []

        # phase_dict: dict[str, list[float]] = {}
        cell_size_unit: str | None = None

        # use condition 1 as x-axis if any else use cond 2
        for r_data in roi_data:
            # r_data = roi_data[roi]

            if amp:
                if r_data.peaks_amplitudes_dec_dff is None:
                    continue

                dp_dict[cond] += r_data.peaks_amplitudes_dec_dff

            elif freq:
                if r_data.dec_dff_frequency is None:
                    continue

                dp_dict[cond].append(r_data.dec_dff_frequency)

            elif iei:
                if r_data.iei is None:
                    continue

                dp_dict[cond] += r_data.iei

            elif cell_size:
                if r_data.cell_size is None:
                    continue

                dp_dict[cond].append(r_data.cell_size)
                cell_size_unit = r_data.cell_size_units

        #     elif synchrony:
        #         if (
        #             r_data.linear_phase is None
        #         ):
        #             continue

        #         phase_dict[roi] = r_data.linear_phase

        # if len(phase_dict.keys()) > 0:
        #     sync_idx = get_connectivity(phase_dict)
        #     dp_list.append(sync_idx)

    data_list, final_cond = [], []
    for cond in cond_ordered:
        if cond in dp_dict:
            data_list.append(dp_dict[cond])
            final_cond.append(cond)

    # violin plot
    # ax.violinplot(data_list)
    # ax.set_xticks([y + 1 for y in range(len(cond_ordered))],
    #   labels=cond_ordered)

    # boxplot
    ax.boxplot(data_list, tick_labels=final_cond, whis=(0, 100))

    _set_axis_labels(ax, amp, freq, iei, cell_size, sync, cell_size_unit)

    widget.figure.tight_layout()

    widget.canvas.draw()


def _set_axis_labels(
    ax: Axes,
    amp: bool,
    freq: bool,
    iei: bool,
    cell_size: bool,
    sync: bool,
    cell_size_unit: str | None,
) -> None:
    """Set axis labels based on the plotted data."""
    if amp:
        ax.set_xlabel("Conditions")
        ax.set_ylabel("Amplitude")
    elif freq:
        ax.set_xlabel("Conditions")
        ax.set_ylabel("Frequency")
    elif iei:
        ax.set_xlabel("Conditions")
        ax.set_ylabel("Inter-event intervals (sec)")
    elif cell_size:
        ax.set_xlabel("Conditions")
        y_axis = f"Cell Size ({cell_size_unit})" if cell_size_unit else "Cell Size"
        ax.set_ylabel(y_axis)
    elif sync:
        ax.set_xlabel("Conditions")
        ax.set_ylabel("Global synchrony")
