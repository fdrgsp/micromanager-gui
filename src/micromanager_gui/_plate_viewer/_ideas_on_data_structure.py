# POSSIBLE DATA STRUCTURE FOR STORING THE DATA


# class Peaks(NamedTuple):
#     peak: int | None = None
#     amplitude: float | None = None
#     raise_time: float | None = None
#     decay_time: float | None = None
#     # ... add whatever other data we need


# class ROIData(NamedTuple):
#     trace: list[float] | None = None
#     dff: list[float] | None = None
#     peaks: list[Peaks] | None = None
#     mean_frequency: float | None = None
#     mean_amplitude: float | None = None
#     # ... add whatever other data we need


# data = {
#     "A1_0000": {   # fov 0000
#             0: ROIData(trace=[list[float]], peaks=[Peaks], mean_frequency=float, mean_amplitude=float, ...),  # roi 0  # noqa
#             1: ROIData(trace=[list[float]], peaks=[Peaks], mean_frequency=float, mean_amplitude=float, ...),  # roi 1  # noqa
#             2: ROIData(trace=[list[float]], peaks=[Peaks], mean_frequency=float, mean_amplitude=float, ...),  # roi 2  # noqa
#             },
#      "A1_0001": {   # fov 0001
#             0: ROIData(trace=[list[float]], peaks=[Peaks], mean_frequency=float, mean_amplitude=float, ...),  # roi 0 # noqa
#             1: ROIData(trace=[list[float]], peaks=[Peaks], mean_frequency=float, mean_amplitude=float, ...),  # roi 1 # noqa
#             2: ROIData(trace=[list[float]], peaks=[Peaks], mean_frequency=float, mean_amplitude=float, ...),  # roi 2 # noqa
#             },
#             ...  # more fovs
#         },
#     "A2_0000": {   # fov 0000
#             0: ROIData(trace=[list[float]], peaks=[Peaks], mean_frequency=float, mean_amplitude=float, ...),  # roi 0 # noqa
#             1: ROIData(trace=[list[float]], peaks=[Peaks], mean_frequency=float, mean_amplitude=float, ...),  # roi 1 # noqa
#             2: ROIData(trace=[list[float]], peaks=[Peaks], mean_frequency=float, mean_amplitude=float, ...),  # roi 2 # noqa
#             },
#     "A2_0001": {   # fov 0001
#             0: ROIData(trace=[list[float]], peaks=[Peaks], mean_frequency=float, mean_amplitude=float, ...),  # roi 0 # noqa
#             1: ROIData(trace=[list[float]], peaks=[Peaks], mean_frequency=float, mean_amplitude=float, ...),  # roi 1 # noqa
#             2: ROIData(trace=[list[float]], peaks=[Peaks], mean_frequency=float, mean_amplitude=float, ...),  # roi 2 # noqa
#             },
#             ...  # more fovs
#         },
#     ...  # more wells
#     }

# the fov (or position) name we get from the metadata is:
#       "A1_0000", "A1_0001", "A1_0002", "A1_0003", ...


# once the 'data' dictionary is filled, we can access the data for a specific
# well, fov, and roi with:

#       data[well_fov] -> will return the dictionary for that well and fov containing
#                        the data for each label (roi).

#       data[well_fov][roi] -> will return the dictionary for that roi containing the
#                          ROIData object for that roi.
#           data[well_fov][roi].trace -> will return the trace data for that roi.
#           data[well_fov][roi].mean_frequency -> will return the mean frequency for that roi.  # noqa
#           data[well_fov][roi].mean_amplitude -> will return the mean amplitude for that roi.  # noqa
#           data[well_fov][roi].peaks -> will return a list of Peaks objects for that roi.  # noqa
#               data[well_fov][roi].peaks[0].peak -> will return the peak timepoint (frame) for the first peak  # noqa
#               data[well_fov][roi].peaks[0].amplitude -> will return the amplitude for the first peak  # noqa
#               data[well_fov][roi].peaks[0].raise_time -> will return the raise time for the first peak  # noqa
#               data[well_fov][roi].peaks[0].decay_time -> will return the decay time for the first peak  # noqa
#               ...


# we can also convert it to a MultiIndex DataFrame
