# from __future__ import annotations

# from typing import TYPE_CHECKING

# import matplotlib.cm as cm
# import matplotlib.colors as mcolors
# import mplcursors
# import numpy as np

# from micromanager_gui._plate_viewer._logger._pv_logger import LOGGER

# if TYPE_CHECKING:
#     from matplotlib.image import AxesImage

#     from micromanager_gui._plate_viewer._graph_widgets import (
#         _SingleWellGraphWidget,
#     )
#     from micromanager_gui._plate_viewer._util import ROIData


# def _get_linear_phase(frames: int, peaks: np.ndarray) -> list[float]:
#     """Calculate the linear phase progression."""
#     if not peaks.any():
#         return [0.0 for _ in range(frames)]
#     peaks_list = [int(peak) for peak in peaks]

#     if any(p < 0 or p >= frames for p in peaks):
#         raise ValueError("All peaks must be within the range of frames.")

#     peaks_list = [int(peak) for peak in peaks]
#     if peaks_list[0] != 0:
#         peaks_list.insert(0, 0)
#     if peaks_list[-1] != (frames - 1):
#         peaks_list.append(frames - 1)

#     phase = [0.0] * frames

#     for k in range(len(peaks_list) - 1):
#         start, end = peaks_list[k], peaks_list[k + 1]

#         if start == end:
#             continue

#         for t in range(start, end):
#             phase[t] = (2 * np.pi) * ((t - start) / (end - start)) + (2 * np.pi * k)

#     phase[frames - 1] = 2 * np.pi * (len(peaks_list) - 1)

#     return phase


# def _get_synchrony_matrix(
#     phase_input: dict[str, list[float]] | np.ndarray,
# ) -> np.ndarray | None:
#     """Compute pairwise synchrony from a phase_dict or phase array."""
#     if isinstance(phase_input, dict):
#         active_rois = list(phase_input.keys())
#         if len(active_rois) < 2:
#             return None
#         try:
#             # convert phase_dict values into NumPy array of shape (#ROIs, #Timepoints)
#             phase_array = np.array(
#                 [phase_input[roi] for roi in active_rois], dtype=np.float32
#             )
#         except ValueError:
#             return None
#     else:
#         # if phase_input is a NumPy array, ensure it is of type float32
#         phase_array = np.asarray(phase_input, dtype=np.float32)
#         if phase_array.shape[0] < 2:
#             return None

#     # compute pairwise phase difference (shape: (#ROIs, #ROIs, #Timepoints))
#     phase_diff = phase_array[:, None, :] - phase_array[None, :, :]
#     # compute Phase-Locking-Value (PLV) matrix directly
#     # e^{jΔφ} = cosΔφ + j sinΔφ ; the magnitude of its time-average is the PLV
#     synchrony_matrix = np.abs(np.mean(np.exp(1j * phase_diff), axis=2))

#     # ensure diagonal elements are exactly 1 (perfect self-synchrony)
#     np.fill_diagonal(synchrony_matrix, 1.0)

#     return synchrony_matrix  # type: ignore


# def _get_synchrony(synchrony_matrix: np.ndarray | None) -> float | None:
#     """Calculate global synchrony score from a synchrony matrix."""
#     if synchrony_matrix is None or synchrony_matrix.size == 0:
#         return None
#     # ensure the matrix is at least 2x2 and square
#     if (
#         synchrony_matrix.shape[0] < 2
#         or synchrony_matrix.shape[0] != synchrony_matrix.shape[1]
#     ):
#         return None

#     # calculate the sum of each row, excluding the diagonal
#     # since diagonal elements are 1, we subtract 1 from each row sum
#     n_rois = synchrony_matrix.shape[0]
#     off_diagonal_sum = np.sum(synchrony_matrix, axis=1) - np.diag(synchrony_matrix)

#     # normalize by the number of off-diagonal elements per row
#     mean_synchrony_per_roi = off_diagonal_sum / (n_rois - 1)

#     # return the median synchrony across all ROIs
#     return float(np.median(mean_synchrony_per_roi))


# def _plot_synchrony_data(
#     widget: _SingleWellGraphWidget,
#     data: dict[str, ROIData],
#     rois: list[int] | None = None,
# ) -> None:
#     """Plot global synchrony."""
#     widget.figure.clear()
#     ax = widget.figure.add_subplot(111)

#     if rois is None:
#         rois = [int(roi) for roi in data if roi.isdigit()]

#     # if less than two rois input, can't calculate synchrony
#     if len(rois) < 2:
#         LOGGER.warning(
#             "Insufficient ROIs selected for synchrony analysis. "
#             "Please select at least two ROIs."
#         )
#         return None

#     phase_dict: dict[str, list[float]] = {}
#     for roi, roi_data in data.items():
#         if int(roi) not in rois:
#             continue
#         if (
#             not roi_data.dec_dff
#             or not roi_data.peaks_dec_dff
#             or len(roi_data.peaks_dec_dff) < 1
#         ):
#             continue
#         frames = len(roi_data.dec_dff)
#         peaks = np.array(roi_data.peaks_dec_dff)
#         phase_dict[roi] = _get_linear_phase(frames, peaks)

#     synchrony_matrix = _get_synchrony_matrix(phase_dict)

#     if synchrony_matrix is None:
#         return None

#     linear_synchrony = _get_synchrony(synchrony_matrix)

#     title = f"Global Synchrony (Median: {linear_synchrony:0.4f})"

#     img = ax.imshow(synchrony_matrix, cmap="viridis", vmin=0, vmax=1)
#     cbar = widget.figure.colorbar(
#         cm.ScalarMappable(cmap="viridis", norm=mcolors.Normalize(vmin=0, vmax=1)),
#         ax=ax,
#     )
#     cbar.set_label("Synchrony index")

#     ax.set_title(title)

#     ax.set_ylabel("ROI")
#     ax.set_yticklabels([])
#     ax.set_yticks([])

#     ax.set_xlabel("ROI")
#     ax.set_xticklabels([])
#     ax.set_xticks([])

#     ax.set_box_aspect(1)

#     active_rois = list(phase_dict.keys())
#     _add_hover_functionality(img, widget, active_rois, synchrony_matrix)
#     widget.figure.tight_layout()
#     widget.canvas.draw()


# def _add_hover_functionality(
#     image: AxesImage,
#     widget: _SingleWellGraphWidget,
#     rois: list[str],
#     synchrony_matrix: np.ndarray,
# ) -> None:
#     """Add hover functionality using mplcursors."""
#     cursor = mplcursors.cursor(image, hover=mplcursors.HoverMode.Transient)

#     @cursor.connect("add")  # type: ignore [misc]
#     def on_add(sel: mplcursors.Selection) -> None:
#         x, y = map(int, np.round(sel.target))  # <-- Snap to nearest pixel center
#         roi_x, roi_y = rois[x], rois[y]

#         sel.annotation.set(
#             text=f"ROI {roi_x} ↔ ROI {roi_y}\nvalue: {synchrony_matrix[y, x]:0.2f}",
#             fontsize=8,
#             color="black",
#         )
#         if roi_x.isdigit() and roi_y.isdigit():
#             widget.roiSelected.emit([roi_x, roi_y])
