from __future__ import annotations

import json
import multiprocessing as mp
from dataclasses import asdict
from multiprocessing import Process
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tifffile
import torch
from cellpose import models
from cellpose.models import CellposeModel
from oasis.functions import deconvolve
from pymmcore_plus._logger import logger
from pymmcore_widgets.useq_widgets._mda_sequence import PYMMCW_METADATA_KEY
from scipy.signal import find_peaks
from tqdm import tqdm

from micromanager_gui._menubar._menubar import EVOKED
from micromanager_gui._plate_viewer._util import (
    COND1,
    COND2,
    ROIData,
    calculate_dff,
    create_stimulation_mask,
    get_cubic_phase,
    get_iei,
    get_linear_phase,
    get_overlap_roi_with_stimulated_area,
)
from micromanager_gui.readers import OMEZarrReader, TensorstoreZarrReader

from ._widgets._mda_widget._save_widget import (
    ALL_EXTENSIONS,
    EXT_TO_WRITER,
    OME_ZARR,
    ZARR_TESNSORSTORE,
)

if TYPE_CHECKING:
    import useq
    from pymmcore_plus import CMMCorePlus

    from micromanager_gui._plate_viewer._plate_map import PlateMapData


CHANNEL = [0, 0]
DIAMETER = 0
CUSTOM = "custom"
CYTO = "cyto3"
ELAPSED_TIME_KEY = "ElapsedTime-ms"
CAMERA_KEY = "camera_metadata"
EXCLUDE_AREA_SIZE_THRESHOLD = 10
STIMULATION_MASK = "stimulation_mask.tif"
STIMULATION_AREA_THRESHOLD = 0.5  # 50%


class SegmentAndAnalyse:
    """Segment neurons."""

    def __init__(self, mmcore: CMMCorePlus):
        self._mmc = mmcore

        self._enabled: bool = False
        self._segment_and_analise_process: Process | None = None
        self._timepoints: int | None = None
        self._model: CellposeModel
        self._stimulation_params: tuple[str, str] | None = None
        self._save_path_and_name: tuple[str, str] | None = None
        self._plate_map_data: dict[str, dict[str, str]] = {}

        # create a multiprocessing Queue
        self._queue: mp.Queue[
            tuple[
                str,  # label_name
                int,  # timepoints
                tuple[str, str],  # save_path_and_name
                CellposeModel,  # model
                tuple[str, str] | None,  # stimulation params
                dict[str, dict[str, str]],  # plate_map_data
            ]
            | None
        ] = mp.Queue()

        self._mmc.mda.events.sequenceStarted.connect(self._on_sequence_started)
        self._mmc.mda.events.frameReady.connect(self._on_frame_ready)
        self._mmc.mda.events.sequenceFinished.connect(self._on_sequence_finished)

    def enable(
        self,
        enable: bool,
        model_type: str,
        model_path: str,
        experiment_type: str,
        stimulation_mask_path: str,
        genotypes: list[PlateMapData],
        treatments: list[PlateMapData],
    ) -> None:
        """Enable or disable the segmentation."""
        self._enabled = enable
        self.set_model(model_type, model_path)
        self.set_stimulation_parameters(experiment_type, stimulation_mask_path)
        self._set_plate_map_data(genotypes, treatments)

    def set_model(self, model_type: str, model_path: str) -> None:
        """Set the cellpose model."""
        use_gpu = torch.cuda.is_available()
        dev = torch.device("cuda" if use_gpu else "cpu")
        if model_type == CUSTOM:
            self._model = CellposeModel(
                pretrained_model=model_path, gpu=use_gpu, device=dev
            )
        else:  # cyto3
            self._model = models.Cellpose(
                gpu=use_gpu, model_type=model_type, device=dev
            )

    def set_stimulation_parameters(
        self, experiment_type: str, stimulation_mask_path: str
    ) -> None:
        self._stimulation_params = (experiment_type, stimulation_mask_path)

    def _set_plate_map_data(
        self, genotypes: list[PlateMapData], treatments: list[PlateMapData]
    ) -> None:
        """Set the plate map data."""
        # update the stored _plate_map_data dict so we have the condition for each well
        # name as the key. e.g.:
        # {"A1": {"condition_1": "condition_1", "condition_2": "condition_2"}}
        self._plate_map_data.clear()
        for data in genotypes:
            self._plate_map_data[data.name] = {COND1: data.condition[0]}
        for data in treatments:
            if data.name in self._plate_map_data:
                self._plate_map_data[data.name][COND2] = data.condition[0]
            else:
                self._plate_map_data[data.name] = {COND2: data.condition[0]}

    def _on_sequence_started(self, sequence: useq.MDASequence) -> None:
        if not self._enabled:
            return

        # get metadata from the sequence
        meta = sequence.metadata.get(PYMMCW_METADATA_KEY, {})
        save_name: str = meta.get("save_name")
        self._save_path_and_name = (meta.get("save_dir"), save_name)

        # continue only if the ZARR_TESNSORSTORE or OME_ZARR
        for ext in ALL_EXTENSIONS:
            if save_name.endswith(ext) and ext in {ZARR_TESNSORSTORE, OME_ZARR}:
                self._enabled = False
                return

        self._timepoints = None
        if sequence.time_plan is not None:
            self._timepoints = sequence.time_plan.num_timepoints() or None

        # create a separate process for segmentation and analysis
        self._segment_and_analise_process = Process(
            target=_segment_and_analyse_process, args=(self._queue,)
        )

        logger.info("SegmentAndAnalyse -> Starting worker...")
        # start the segmentation process
        self._segment_and_analise_process.start()

    def _on_frame_ready(self, image: np.ndarray, event: useq.MDAEvent) -> None:
        if not self._enabled or self._save_path_and_name is None:
            return

        if (t_index := event.index.get("t")) is None or self._timepoints is None:
            return

        # send it to the segmentation and analysis process
        if t_index == self._timepoints - 1:
            # create the labels file name
            p_idx = event.index.get("p", 0)

            pos_name = (
                event.pos_name
                if event.pos_name is not None
                else f"pos_{str(p_idx).zfill(4)}"
            )
            label_name = f"{pos_name}_p{p_idx}"

            # send the max_proj image to the segmentation process
            self._queue.put(
                (
                    label_name,
                    self._timepoints,
                    self._save_path_and_name,
                    self._model,
                    self._stimulation_params,
                    self._plate_map_data,
                )
            )

            logger.info(
                f"SegmentAndAnalyse -> Sending info to segment and analyse {pos_name}."
            )

    def _on_sequence_finished(self, sequence: useq.MDASequence) -> None:
        if not self._enabled:
            return

        # add the None to the queue to exit the process
        self._queue.put(None)
        # wait for the process to finish before exiting
        if self._segment_and_analise_process is not None:
            self._segment_and_analise_process.join()
        self._segment_and_analise_process = None
        logger.info("SegmentAndAnalyse -> Segmentation worker stopped.")


# this must not be part of the SegmentAndAnalyse class
def _segment_and_analyse_process(queue: mp.Queue) -> None:
    """Segmentation worker running in a separate process."""
    # to store the datastore opened in the process so we don't have to reopen it
    datastore_cache: dict[Path, TensorstoreZarrReader | OMEZarrReader] = {}
    # exit the process if the queue is empty (None)
    while (args := queue.get()) is not None:
        (
            label_name,
            timepoints,
            save_path_and_name,
            model,
            stimulation_params,
            plate_map_data,
        ) = args
        save_path, save_name = save_path_and_name

        # get the datastore path
        datastore_path = Path(save_path) / save_name

        # remove the extension from the save_name and get the writer
        writer = ""
        for ext in ALL_EXTENSIONS:
            if save_name.endswith(ext):
                save_name = save_name[: -len(ext)]
                writer = EXT_TO_WRITER[ext]
                break

        # if datastore is already opened, reuse it
        if datastore_path in datastore_cache:
            datastore = datastore_cache[datastore_path]
        else:
            # open the datastore and store it in cache
            if writer == ZARR_TESNSORSTORE:
                datastore = TensorstoreZarrReader(datastore_path)
            elif writer == OME_ZARR:
                datastore = OMEZarrReader(datastore_path)
            else:
                return  # unsupported file format
            datastore_cache[datastore_path] = datastore  # cache the datastore

        # call the segmentation function with the cached datastore
        _segment_and_analyse(
            save_path,
            save_name,
            label_name,
            timepoints,
            datastore,
            model,
            stimulation_params,
            plate_map_data,
        )
    # clear the cache
    datastore_cache.clear()


def _segment_and_analyse(
    save_path: str,
    save_name: str,
    label_name: str,
    timepoints: int,
    datastore: TensorstoreZarrReader | OMEZarrReader,
    model: CellposeModel,
    stimulation_params: tuple[str, str],
    plate_map_data: dict[str, dict[str, str]],
) -> None:
    """Segment the image."""
    logger.info(f"SegmentAndAnalyse -> --------{label_name}--------")

    # get the experiment type and stimulation mask path if any
    experiment_type, stimulation_mask_path = stimulation_params

    # create save directories if they don't exist
    save_dir_labels = Path(save_path) / f"{save_name}_labels"
    save_dir_labels.mkdir(parents=True, exist_ok=True)
    save_dir_analysis = Path(save_path) / f"{save_name}_analysis"
    save_dir_analysis.mkdir(parents=True, exist_ok=True)

    # get position index
    p_idx = int(label_name.split("_")[-1][1:])  # extract 'p23' -> 23

    # get data and metadata from the datastore for the position index
    data, meta = datastore.isel(p=p_idx, metadata=True)

    # get the position index from label_name
    p_idx = int(label_name.split("_")[-1][1:])  # a1_0020_p23 -> p23-> 23

    # get data and metadata from the datastore for the position index
    data, meta = datastore.isel(p=p_idx, metadata=True)

    # SEGMENTATION - CELLPOSE --------------------------------------------------------
    logger.info(f"SegmentAndAnalyse -> Segmenting image: {label_name}...")
    # max projection from half to the end of the stack
    data_half_to_end = data[data.shape[0] // 2 :, :, :]
    max_proj = data_half_to_end.max(axis=0)

    # run cellpose
    output = model.eval(max_proj, diameter=DIAMETER, channels=CHANNEL)
    labels = output[0]

    # save to disk
    logger.info(
        f"SegmentAndAnalyse -> Saving labels: {save_dir_labels}/{label_name}.tif"
    )
    tifffile.imwrite(save_dir_labels / f"{label_name}.tif", labels)

    # ANALYSIS -----------------------------------------------------------------------
    stimulated_area_mask: np.ndarray | None = None
    if experiment_type == EVOKED and stimulation_mask_path:
        logger.info(f"SegmentAndAnalyse -> Creating stimulation mask for {label_name}")
        stimulated_area_mask = create_stimulation_mask(stimulation_mask_path)
        stim_mask_path = save_dir_labels / STIMULATION_MASK
        tifffile.imwrite(str(stim_mask_path), stimulated_area_mask)
        logger.info(f"SegmentAndAnalyse -> Stimulation mask saved for {label_name}")

    # extract traces data
    logger.info(f"SegmentAndAnalyse -> Extracting traces data for {label_name}...")
    analysis_data = _extract_traces_data(
        label_name, timepoints, data, meta, labels, stimulated_area_mask, plate_map_data
    )
    if analysis_data is None:
        return

    # save the analysis data to disk
    analysis_data_path = save_dir_analysis / f"{label_name}.json"
    with analysis_data_path.open("w") as f:
        json.dump(
            analysis_data,
            f,
            default=lambda o: asdict(o) if isinstance(o, ROIData) else o,
            indent=2,
        )
    logger.info(f"SegmentAndAnalyse -> Analysis data saved for {label_name}")


def _extract_traces_data(
    label_name: str,
    timepoints: int,
    data: np.ndarray,
    meta: list[dict],
    labels: np.ndarray,
    stimulated_area_mask: np.ndarray | None,
    plate_map_data: dict[str, dict[str, str]],
) -> dict[str, ROIData] | None:
    """Extract traces data."""
    # the "Event" key was used in the old metadata format
    event_key = "mda_event" if "mda_event" in meta[0] else "Event"

    # get the elapsed time from the metadata to calculate the total time in seconds
    elapsed_time_list = _get_elapsed_time_list(meta)

    # get the exposure time from the metadata
    exp_time = meta[0][event_key].get("exposure", 0.0)

    # get the total time in seconds for the recording
    tot_time_sec = _calculate_total_time(elapsed_time_list, exp_time, timepoints)

    # create a dict with the labels as keys and the masks as values
    labels_masks = _create_label_masks_dict(labels)

    analysis_data: dict = {}

    for label_value, label_mask in tqdm(
        labels_masks.items(), desc=f"Extracting Traces from Well {label_name}"
    ):
        # calculate the mean trace for the roi
        masked_data = data[:, label_mask]

        # get the size of the roi in µm or px if µm is not available
        roi_size_pixel = masked_data.shape[1]  # area
        px_size = meta[0].get("PixelSizeUm", None)
        # calculate the size of the roi in µm if px_size is available or not 0,
        # otherwise use the size is in pixels
        roi_size = roi_size_pixel * px_size if px_size else roi_size_pixel

        # exclude small rois, might not be necessary if trained cellpose performs
        # better
        if px_size and roi_size < EXCLUDE_AREA_SIZE_THRESHOLD:
            return None

        # check if the roi is stimulated
        roi_stimulation_overlap_ratio = 0.0
        if stimulated_area_mask is not None:
            roi_stimulation_overlap_ratio = get_overlap_roi_with_stimulated_area(
                stimulated_area_mask, label_mask
            )

        # compute the mean for each frame
        roi_trace: np.ndarray = masked_data.mean(axis=1)

        # calculate the dff of the roi trace
        dff: np.ndarray = calculate_dff(roi_trace, window=10, plot=False)

        # deconvolve the dff trace
        dec_dff, spikes, _, _, _ = deconvolve(dff, penalty=1)

        # get the prominence to find peaks in the deconvolved trace
        # -	Step 1: np.median(dff) -> The median of the dataset dff is computed. The
        # median is the “middle” value of the dataset when sorted, which is robust
        # to outliers (unlike the mean).
        # -	Step 2: np.abs(dff - np.median(dff)) -> The absolute deviation of each
        # value in dff from the median is calculated. This measures how far each
        # value is from the central point (the median).
        # -	Step 3: np.median(...) -> The median of the absolute deviations is
        # computed. This gives the Median Absolute Deviation (MAD), which is a
        # robust measure of the spread of the data. Unlike standard deviation, the
        # MAD is not influenced by extreme outliers.
        # -	Step 4: Division by 0.6745 -> The constant 0.6745 rescales the MAD to
        # make it comparable to the standard deviation if the data follows a normal
        # (Gaussian) distribution. Specifically: for a normal distribution,
        # MAD ≈ 0.6745 * standard deviation. Dividing by 0.6745 converts the MAD
        # into an estimate of the standard deviation.
        noise_level_dec_dff = np.median(np.abs(dec_dff - np.median(dec_dff))) / 0.6745
        peaks_prominence_dec_dff = noise_level_dec_dff * 2

        # find peaks in the deconvolved trace
        peaks_dec_dff, _ = find_peaks(dec_dff, prominence=peaks_prominence_dec_dff)

        # get the amplitudes of the peaks in the dec_dff trace
        peaks_amplitudes_dec_dff = [dec_dff[p] for p in peaks_dec_dff]

        # calculate the frequency of the peaks in the dec_dff trace
        frequency = len(peaks_dec_dff) / tot_time_sec if tot_time_sec else 0.0

        # get the conditions for the well
        condition_1, condition_2 = _get_conditions(label_name, plate_map_data)

        # get the linear and cubic phase of the peaks in the dec_dff trace
        linear_phase, cubic_phase = [], []
        if len(peaks_dec_dff) > 0:
            linear_phase = get_linear_phase(timepoints, peaks_dec_dff)
            cubic_phase = get_cubic_phase(timepoints, peaks_dec_dff)

        # if the elapsed time is not available or for any reason is different from
        # the number of timepoints, set it as list of timepoints every exp_time
        if len(elapsed_time_list) != timepoints:
            elapsed_time_list = [i * exp_time for i in range(timepoints)]

        # calculate the inter-event interval (IEI) of the peaks in the dec_dff trace
        iei = get_iei(peaks_dec_dff, elapsed_time_list)

        # store the data to the analysis dict as ROIData
        analysis_data[str(label_value)] = ROIData(
            well_fov_position=label_name,
            raw_trace=roi_trace.tolist(),  # type: ignore
            dff=dff.tolist(),  # type: ignore
            dec_dff=dec_dff.tolist(),
            peaks_dec_dff=peaks_dec_dff.tolist(),
            peaks_amplitudes_dec_dff=peaks_amplitudes_dec_dff,
            peaks_prominence_dec_dff=peaks_prominence_dec_dff,
            dec_dff_frequency=frequency,
            inferred_spikes=spikes.tolist(),
            cell_size=roi_size,
            cell_size_units="µm" if px_size is not None else "pixel",
            condition_1=condition_1,
            condition_2=condition_2,
            total_recording_time_in_sec=tot_time_sec,
            active=len(peaks_dec_dff) > 0,
            linear_phase=linear_phase,
            cubic_phase=cubic_phase,
            iei=iei,
            stimulated=roi_stimulation_overlap_ratio > STIMULATION_AREA_THRESHOLD,
        )

    return analysis_data


def _get_elapsed_time_list(meta: list[dict]) -> list[float]:
    elapsed_time_list: list[float] = []
    # get the elapsed time for each timepoint to calculate tot_time_sec
    if (cam_key := CAMERA_KEY) in meta[0]:  # new metadata format
        for m in meta:
            et = m[cam_key].get(ELAPSED_TIME_KEY)
            if et is not None:
                elapsed_time_list.append(float(et))
    else:  # old metadata format
        for m in meta:
            et = m.get(ELAPSED_TIME_KEY)
            if et is not None:
                elapsed_time_list.append(float(et))
    return elapsed_time_list


def _calculate_total_time(
    elapsed_time_list: list[float],
    exp_time: float,
    timepoints: int,
) -> float:
    """Calculate total time in seconds for the recording."""
    # if the len of elapsed time is not equal to the number of timepoints,
    # use exposure time and the number of timepoints to calculate tot_time_sec
    if len(elapsed_time_list) != timepoints:
        tot_time_sec = exp_time * timepoints / 1000
    # otherwise, calculate the total time in seconds using the elapsed time.
    # nOTE: adding the exposure time to consider the first frame
    else:
        tot_time_sec = (elapsed_time_list[-1] - elapsed_time_list[0] + exp_time) / 1000
    return tot_time_sec


def _create_label_masks_dict(labels: np.ndarray) -> dict:
    """Create masks for each label in the labels image."""
    # get the range of labels and remove the background (0)
    labels_range = np.unique(labels[labels != 0])
    return {label_value: (labels == label_value) for label_value in labels_range}


def _get_conditions(
    label_name: str, plate_map_data: dict[str, dict[str, str]]
) -> tuple[str | None, str | None]:
    """Get the conditions for the well if any."""
    condition_1 = condition_2 = None
    if plate_map_data:
        well_name = label_name.split("_")[0]
        if well_name in plate_map_data:
            condition_1 = plate_map_data[well_name].get(COND1)
            condition_2 = plate_map_data[well_name].get(COND2)
        else:
            condition_1 = condition_2 = None
    return condition_1, condition_2
