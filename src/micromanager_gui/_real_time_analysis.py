from __future__ import annotations

import bisect
import json
import multiprocessing as mp
import re
from dataclasses import asdict
from multiprocessing import Process
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

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
    MWCM,
    ROIData,
    calculate_dff,
    create_stimulation_mask,
    get_iei,
    get_linear_phase,
    get_overlap_roi_with_stimulated_area,
)
from micromanager_gui._widgets._mda_widget._mda_widget import ANALYSIS
from micromanager_gui._widgets._mda_widget._real_time_analysis_wdg import (
    AnalysisParameters,
    SegmentationParameters,
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
    from pymmcore_plus.metadata import FrameMetaV1

    from micromanager_gui._plate_viewer._plate_map import PlateMapData


CHANNEL = [0, 0]
DIAMETER = 0
CUSTOM = "custom"
CYTO = "cyto3"
ELAPSED_TIME_KEY = "ElapsedTime-ms"
CAMERA_KEY = "camera_metadata"
EVENT_KEY = "mda_event"
EXCLUDE_AREA_SIZE_THRESHOLD = 10
STIMULATION_MASK = "stimulation_mask.tif"
STIMULATION_AREA_THRESHOLD = 0.1  # 10%


class RealTimeAnalysis:
    """Run segmentation and analysis in real-time."""

    def __init__(self, mmcore: CMMCorePlus):
        self._mmc = mmcore

        self._enabled_segmentation: bool = False
        self._enabled_analysis: bool = False

        self._segment_and_analise_process: Process | None = None
        self._timepoints: int | None = None
        self._model: CellposeModel
        self._evoked_exp_params: tuple[str, str, str] | None = None
        self._save_info: tuple[str, str] | None = None
        self._plate_map_data: dict[str, dict[str, str]] = {}
        self._meta: list[FrameMetaV1] = []
        self._min_peaks_height: float = 0.0

        # create a multiprocessing Queue
        self._queue: mp.Queue[
            tuple[
                str,  # label_name
                tuple[str, str],  # save_info
                CellposeModel,  # model
            ]
            | tuple[
                str,  # label_name
                tuple[str, str],  # save_info
                tuple[float, list[float], float, int],  # meta_info
                tuple[
                    CellposeModel,  # model (analysis_info)
                    float,  # min_peaks_height (analysis_info)
                    tuple[str, str, str],  # evoked_params (analysis_info)
                    dict[str, Any] | None,  # evoked_meta (analysis_info)
                    dict[str, dict[str, str]],  # plate_map_data (analysis_info)
                ],
            ]
            | None
        ] = mp.Queue()

        self._mmc.mda.events.sequenceStarted.connect(self._on_sequence_started)
        self._mmc.mda.events.frameReady.connect(self._on_frame_ready)
        self._mmc.mda.events.sequenceFinished.connect(self._on_sequence_finished)

    def set_experiment_type_parameters(
        self, experiment_type: str, stimulation_mask_path: str, led_power_equation: str
    ) -> None:
        """Set the stimulation parameters."""
        self._evoked_exp_params = (
            experiment_type,
            stimulation_mask_path,
            led_power_equation,
        )

    def set_model(self, model_type: str, model_path: str) -> None:
        """Set the cellpose model."""
        if not model_type:
            return
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
        # get metadata from the sequence
        meta = sequence.metadata.get(PYMMCW_METADATA_KEY, {})

        # do not run the real-time analysis if the analysis is not set in the metadata
        # parameters: {
        #   'segmentation': SegmentationParameters | None,
        #   'analysis': AnalysisParameters | None
        # } or None if ANALYSIS key is not set in the metadata
        parameters: dict[str, SegmentationParameters | AnalysisParameters] | None = (
            meta.get(ANALYSIS, None)
        )

        # if "analysis" is not set in the metadata, disable the real-time analysis
        if parameters is None:
            self._enabled_segmentation, self._enabled_analysis = (False, False)
            return

        segmentation_params = cast(
            SegmentationParameters | None, parameters.get("segmentation")
        )
        analysis_params = cast(AnalysisParameters | None, parameters.get("analysis"))
        # if "segmentation" is not set in the metadata, disable the real-time analysis
        # because analysis cannot be run without segmentation
        if segmentation_params is None:
            self._enabled_segmentation, self._enabled_analysis = (False, False)
            return

        self._enabled_segmentation = segmentation_params is not None  # True
        self._enabled_analysis = analysis_params is not None

        # disable if save_name or save_path is not available
        save_name: str = meta.get("save_name", "")
        save_path: str = meta.get("save_dir", "")
        if not save_name or not save_path:
            self._save_info = None
            self._enabled_segmentation = False
            self._enabled_analysis = False
            logger.error(
                "SegmentAndAnalyse -> Save path or name not available! "
                "Disabling real-time analysis since the save path and name are "
                "required to run the real-time analysis."
            )

        # continue only if the ZARR_TESNSORSTORE or OME_ZARR
        for ext in ALL_EXTENSIONS:
            if save_name.endswith(ext) and ext in {ZARR_TESNSORSTORE, OME_ZARR}:
                self._enabled_segmentation = False
                self._enabled_analysis = False
                logger.error(
                    f"SegmentAndAnalyse -> Unsupported file format: {ext}. "
                    "Disabling real-time analysis since only ZARR_TESNSORSTORE "
                    "and OME_ZARR are supported."
                )
                return

        # set the parameters for the analysis
        self._set_parameters(segmentation_params, analysis_params)
        self._save_info = (save_path, save_name)

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

    def _set_parameters(
        self,
        segmentation_params: SegmentationParameters,
        analysis_params: AnalysisParameters | None,
    ) -> None:
        """Set the parameters for the analysis."""
        self.set_model(
            segmentation_params.get("model_type", ""),
            segmentation_params.get("model_path", ""),
        )

        if analysis_params is not None:
            self.set_experiment_type_parameters(
                analysis_params.get("experiment_type", ""),
                analysis_params.get("stimulation_mask_path", ""),
                analysis_params.get("led_power_equation", ""),
            )
            self._set_plate_map_data(
                analysis_params.get("genotypes", []),  # type: ignore
                analysis_params.get("treatments", []),  # type: ignore
            )
            self._min_peaks_height = analysis_params.get("min_peaks_height", 0.0)

    def _on_frame_ready(
        self, image: np.ndarray, event: useq.MDAEvent, meta: FrameMetaV1
    ) -> None:
        if not self._enabled_segmentation or self._save_info is None:
            return

        if (t_index := event.index.get("t")) is None or self._timepoints is None:
            return

        self._meta.append(meta)

        # send it to the segmentation and analysis process
        if t_index == self._timepoints - 1:
            self._update_queue(event)
            self._meta.clear()

    def _update_queue(self, event: useq.MDAEvent) -> None:
        self._save_info = cast(tuple[str, str], self._save_info)

        # create the labels file name
        p_idx = event.index.get("p", 0)

        pos_name = (
            event.pos_name
            if event.pos_name is not None
            else f"pos_{str(p_idx).zfill(4)}"
        )
        label_name = f"{pos_name}_p{p_idx}"

        if self._enabled_analysis is None:
            # send the max_proj image to the segmentation process
            self._queue.put(label_name, self._save_info, self._model)
            logger.info(f"SegmentAndAnalyse -> Sending to segment {label_name} image.")
            return

        meta_info = (
            event.exposure or self._mmc.getExposure(),
            _get_elapsed_time_list(self._meta),
            self._mmc.getPixelSizeUm(),
            cast(int, self._timepoints),
        )

        evoked_meta: dict[str, Any] | None = None
        if (seq := event.sequence) is not None:
            evoked_meta = seq.metadata.get(PYMMCW_METADATA_KEY, {}).get(
                "stimulation", None
            )

        parameters = (
            self._model,
            self._min_peaks_height,
            cast(tuple[str, str, str], self._evoked_exp_params),
            evoked_meta,
            self._plate_map_data,
        )

        # send the max_proj image to the segmentation process
        self._queue.put((label_name, self._save_info, meta_info, parameters))

        logger.info(
            f"SegmentAndAnalyse -> Sending info to segment and analyse {label_name}."
        )

    def _on_sequence_finished(self, sequence: useq.MDASequence) -> None:
        if not self._enabled_segmentation:
            return

        # add the None to the queue to exit the process
        self._queue.put(None)
        # wait for the process to finish before exiting
        if self._segment_and_analise_process is not None:
            self._segment_and_analise_process.join()
        self._segment_and_analise_process = None
        logger.info("SegmentAndAnalyse -> Segmentation worker stopped.")


def _get_datastore(
    datastore_path: Path, save_path: str, save_name: str
) -> TensorstoreZarrReader | OMEZarrReader | None:
    """Get the datastore from the path."""
    datastore_cache: dict[Path, TensorstoreZarrReader | OMEZarrReader] = {}
    # remove the extension from the save_name and get the writer
    # if datastore is already opened, reuse it
    if datastore_path in datastore_cache:
        datastore = datastore_cache[datastore_path]
    else:
        # remove the extension from the save_name and get the writer
        writer = ""
        for ext in ALL_EXTENSIONS:
            if save_name.endswith(ext):
                save_name = save_name[: -len(ext)]
                writer = EXT_TO_WRITER[ext]
                break
        # open the datastore and store it in cache
        if writer == ZARR_TESNSORSTORE:
            datastore = TensorstoreZarrReader(datastore_path)
        elif writer == OME_ZARR:
            datastore = OMEZarrReader(datastore_path)
        else:
            return None
        datastore_cache[datastore_path] = datastore  # cache the datastore
    return datastore


# this must not be part of the SegmentAndAnalyse class
def _segment_and_analyse_process(queue: mp.Queue) -> None:
    """Segmentation worker running in a separate process."""
    # to store the datastore opened in the process so we don't have to reopen it
    datastore_cache: dict[Path, TensorstoreZarrReader | OMEZarrReader] = {}
    # exit the process if the queue is empty (None)
    while (args := queue.get()) is not None:
        if len(args) == 3:  # segmentation only
            label_name, (save_path, save_name), model = args
            # get the datastore path
            datastore_path = Path(save_path) / save_name
            datastore = _get_datastore(datastore_path, save_path, save_name)
            if datastore is None:
                logger.error(
                    f"SegmentAndAnalyse -> Datastore not found for {label_name}."
                )
                return
            _segment(label_name, datastore, save_path, save_name, model)
        else:
            # get the arguments
            label_name, save_info, meta_info, analysis_info = args
            save_path, save_name = save_info

            # get the datastore path
            datastore_path = Path(save_path) / save_name
            datastore = _get_datastore(datastore_path, save_path, save_name)
            if datastore is None:
                logger.error(
                    f"SegmentAndAnalyse -> Datastore not found for {label_name}."
                )
                return
            # call the segmentation function with the cached datastore
            _segment_and_analyse(
                label_name, datastore, save_info, analysis_info, meta_info
            )
    # clear the cache
    datastore_cache.clear()


def _segment(
    label_name: str,
    datastore: TensorstoreZarrReader | OMEZarrReader,
    save_path: str,
    save_name: str,
    model: CellposeModel,
) -> np.ndarray:
    """Segment the image."""
    save_dir_labels = Path(save_path) / f"{save_name}_labels"
    save_dir_labels.mkdir(parents=True, exist_ok=True)

    # get the position index from label_name
    p_idx = int(label_name.split("_")[-1][1:])  # a1_0020_p23 -> p23-> 23

    # get data and metadata from the datastore for the position index
    # NOTE: cannot use datastore.isel(p=p_idx) because metadata are not yet stored
    data = datastore.store[p_idx].read().result().squeeze()

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

    return labels  # type: ignore


def _segment_and_analyse(
    label_name: str,
    datastore: TensorstoreZarrReader | OMEZarrReader,
    save_info: tuple[str, str],
    analysis_info: tuple[
        CellposeModel,
        float,
        tuple[str, str, str],
        dict[str, Any] | None,
        dict[str, dict[str, str]],
    ],
    meta_info: tuple[float, list[float], float, int],
) -> None:
    """Segment the image and run the analysis."""
    save_path, save_name = save_info
    model, min_peaks_height, evoked_params, evoked_meta, plate_map_data = analysis_info

    labels = _segment(label_name, datastore, save_path, save_name, model)
    save_dir_labels = Path(save_path) / f"{save_name}_labels"

    # create save directories if they don't exist
    save_dir_analysis = Path(save_path) / f"{save_name}_analysis"
    save_dir_analysis.mkdir(parents=True, exist_ok=True)

    # get the position index from label_name
    p_idx = int(label_name.split("_")[-1][1:])  # a1_0020_p23 -> p23-> 23

    # get data and metadata from the datastore for the position index
    # NOTE: cannot use datastore.isel(p=p_idx) because metadata are not yet stored
    data = datastore.store[p_idx].read().result().squeeze()

    # ANALYSIS -----------------------------------------------------------------------
    # get the experiment type and stimulation mask path if any
    experiment_type, stimulation_mask_path, led_power_equation = evoked_params
    stimulated_area_mask: np.ndarray | None = None
    evoked_experiment: bool = experiment_type == EVOKED
    if evoked_meta and evoked_experiment and stimulation_mask_path:
        logger.info(f"SegmentAndAnalyse -> Creating stimulation mask for {label_name}")
        stimulated_area_mask = create_stimulation_mask(stimulation_mask_path)
        stim_mask_path = save_dir_labels / STIMULATION_MASK
        tifffile.imwrite(str(stim_mask_path), stimulated_area_mask)
        logger.info(f"SegmentAndAnalyse -> Stimulation mask saved for {label_name}")

    # extract traces data
    logger.info(f"SegmentAndAnalyse -> Extracting traces data for {label_name}...")
    analysis_data = _extract_traces_data(
        label_name,
        cast(np.ndarray, data),
        labels,
        min_peaks_height,
        evoked_experiment,
        stimulated_area_mask,
        evoked_meta,
        led_power_equation,
        plate_map_data,
        meta_info,
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
    data: np.ndarray,
    labels: np.ndarray,
    min_peaks_height: float,
    evoked_exp: bool,
    stimulated_area_mask: np.ndarray | None,
    evoked_meta: dict[str, Any] | None,
    led_power_equation: str,
    plate_map_data: dict[str, dict[str, str]],
    meta_info: tuple[float, list[float], float, int],
) -> dict[str, ROIData] | None:
    """Extract traces data."""
    exp_time, elapsed_time_list, pixel_size, timepoints = meta_info
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
        # calculate the size of the roi in µm if px_size is available or not 0,
        # otherwise use the size is in pixels
        roi_size = roi_size_pixel * (pixel_size**2) if pixel_size else roi_size_pixel

        # exclude small rois, might not be necessary if trained cellpose performs
        # better
        if pixel_size and roi_size < EXCLUDE_AREA_SIZE_THRESHOLD:
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
        peaks_prominence_dec_dff = noise_level_dec_dff  # * 2

        # find peaks in the deconvolved trace
        peaks_dec_dff, _ = find_peaks(
            dec_dff, prominence=peaks_prominence_dec_dff, height=min_peaks_height
        )

        # get the amplitudes of the peaks in the dec_dff trace
        peaks_amplitudes_dec_dff = [dec_dff[p] for p in peaks_dec_dff]

        # check if the roi is stimulated
        is_roi_stimulated = roi_stimulation_overlap_ratio > STIMULATION_AREA_THRESHOLD

        # to store the amplitudes as dict: {power_pulselength: [amplitude]}
        amplitudes_stimulated_peaks: dict[str, list[float]] = {}
        amplitudes_non_stimulated_peaks: dict[str, list[float]] = {}

        # if the experiment is evoked, get the amplitudes of the stimulated peaks
        if evoked_exp and evoked_meta is not None and len(peaks_dec_dff) > 0:
            # get the stimulation info from the metadata (if any)
            amplitudes_stimulated_peaks, amplitudes_non_stimulated_peaks = (
                _update_stim_vs_non_stim(
                    evoked_meta,
                    dec_dff,
                    peaks_dec_dff,
                    is_roi_stimulated,
                    led_power_equation,
                )
            )

        # calculate the frequency of the peaks in the dec_dff trace
        frequency = (
            len(peaks_dec_dff) / tot_time_sec
            if tot_time_sec and len(peaks_dec_dff) > 0
            else None
        )

        # get the conditions for the well
        condition_1, condition_2 = _get_conditions(label_name, plate_map_data)

        # calculate the instantaneous phase of the peaks in the dec_dff trace
        instantaneous_phase = (
            get_linear_phase(timepoints, peaks_dec_dff)
            if len(peaks_dec_dff) > 0
            else None
        )

        # if the elapsed time is not available or for any reason is different from
        # the number of timepoints, set it as list of timepoints every exp_time
        if len(elapsed_time_list) != timepoints:
            elapsed_time_list = [i * exp_time for i in range(timepoints)]

        # calculate the inter-event interval (IEI) of the peaks in the dec_dff trace
        iei = get_iei(peaks_dec_dff, elapsed_time_list)

        # store the data to the analysis dict as ROIData
        analysis_data[str(label_value)] = ROIData(
            well_fov_position=label_name,
            raw_trace=roi_trace.tolist(),
            dff=dff.tolist(),
            dec_dff=dec_dff.tolist(),
            peaks_dec_dff=peaks_dec_dff.tolist(),
            peaks_amplitudes_dec_dff=peaks_amplitudes_dec_dff,
            peaks_prominence_dec_dff=peaks_prominence_dec_dff,
            dec_dff_frequency=frequency or None,
            inferred_spikes=spikes.tolist(),
            cell_size=roi_size,
            cell_size_units="µm" if pixel_size is not None else "pixel",
            condition_1=condition_1,
            condition_2=condition_2,
            total_recording_time_in_sec=tot_time_sec,
            active=len(peaks_dec_dff) > 0,
            instantaneous_phase=instantaneous_phase,
            iei=iei,
            evoked_experiment=evoked_exp,
            stimulated=is_roi_stimulated,
            amplitudes_stimulated_peaks=amplitudes_stimulated_peaks or None,
            amplitudes_non_stimulated_peaks=amplitudes_non_stimulated_peaks or None,
        )

    return analysis_data


def _get_elapsed_time_list(meta: list[FrameMetaV1]) -> list[float]:
    elapsed_time_list: list[float] = []
    # get the elapsed time for each timepoint to calculate tot_time_sec
    if CAMERA_KEY in meta[0]:
        for m in meta:
            et = m[CAMERA_KEY].get(ELAPSED_TIME_KEY)  # type: ignore
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


def linear_equation_from_str(equation: str) -> Callable | None:
    # Match format: y = m*x + q (allowing optional spaces and + or - for q)
    if not equation:
        return None

    match = re.match(
        r"y\s*=\s*([+-]?\d*\.?\d+)\s*\*\s*x\s*([+-]\s*\d*\.?\d+)", equation
    )
    if not match:
        msg = (
            "Invalid equation format! Should be in the form: y = m * x + q\n"
            "Using values from the metadata."
        )
        logger.error(msg)
        return None

    m = float(match[1])
    q = float(match[2].replace(" ", ""))
    return lambda x: m * x + q


def _update_stim_vs_non_stim(
    evoked_experiment_meta: dict[str, Any],
    dec_dff: np.ndarray,
    peaks_dec_dff: np.ndarray,
    is_roi_stimulated: bool,
    led_power_equation: str,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Update the stimulated and non-stimulated peaks amplitude dict."""
    # to store the amplitudes as dict: {power_pulselength: [amplitude]}
    amplitudes_stimulated_peaks: dict[str, list[float]] = {}
    amplitudes_non_stimulated_peaks: dict[str, list[float]] = {}

    frames_and_powers = evoked_experiment_meta.get("pulse_on_frame", {})
    sorted_peaks_dec_dff = list(sorted(peaks_dec_dff))  # noqa: C413

    eq: Callable | None = None
    if led_power_equation:
        eq = linear_equation_from_str(led_power_equation)

    for frame, power in frames_and_powers.items():
        stim_frame = int(frame) + 1
        # find index of first peak >= stim_frame.
        i = bisect.bisect_left(sorted_peaks_dec_dff, stim_frame)
        # Note that if the frame is not found, bisect_left returns the index
        # where it would be inserted and so the max index + 1. We need to check
        # if the index is valid and, if not, skip it.
        if i >= len(sorted_peaks_dec_dff):
            continue
        peak_idx = sorted_peaks_dec_dff[i]
        # check if the peak is on the stimulation frame or in the next 5 frames
        if peak_idx >= stim_frame and peak_idx <= stim_frame + 5:
            amplitude = dec_dff[peak_idx]
            pulse_len = evoked_experiment_meta.get("led_pulse_duration", "unknown")
            if eq is not None:
                power = eq(power)
                power = f"{power:.3f}{MWCM}"
            col = f"{power}_{pulse_len}"
            if is_roi_stimulated:
                amplitudes_stimulated_peaks.setdefault(col, []).append(amplitude)
            else:
                amplitudes_non_stimulated_peaks.setdefault(col, []).append(amplitude)
    return amplitudes_stimulated_peaks, amplitudes_non_stimulated_peaks
