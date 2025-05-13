from __future__ import annotations

from typing import TYPE_CHECKING

import mplcursors

if TYPE_CHECKING:
    from micromanager_gui._plate_viewer._graph_widgets import _MultilWellGraphWidget
    from micromanager_gui._plate_viewer._util import ROIData


COUNT_INCREMENT = 1
DEFAULT_COLOR = "gray"
STIMULATED_COLOR = "green"
NON_STIMULATED_COLOR = "magenta"


def _plot_multi_well_data(
    widget: _MultilWellGraphWidget,
    data: dict[str, dict[str, ROIData]],
    positions: list[int] | None = None,
    amp: bool = False,
    freq: bool = False,
    iei: bool = False,
) -> None:
    """Plot the multi-well data."""
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    well_count: int = 0
    wells_names: list[str] = []
    # loop over the data and plot the data
    for well_and_fov, roi_data in data.items():
        if positions is not None and int(well_and_fov) not in positions:
            continue

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
                ax.set_ylabel("Frequency (Hz)")

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
                ax.set_ylabel("Frequency (Hz)")

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

    # set ticks labels as well names
    if (amp or freq or iei) and not (amp and freq):
        ax.set_xticks(range(1, len(wells_names) + 1))
        ax.set_xticklabels(wells_names)

    widget.figure.tight_layout()

    # add hover functionality to display the well position
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        sel.annotation.set(text=sel.artist.get_label(), fontsize=8, color="black")

    widget.canvas.draw()
