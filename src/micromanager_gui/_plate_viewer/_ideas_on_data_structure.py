# POSSIBLE DATA STRUCTURE FOR STORING THE DATA


# class ROIData(NamedTuple):
#     trace: list[float]
#     peaks: list[Peaks]
#     mean_frequency: float
#     mean_amplitude: float
#     ... add whatever other data we need


# class Peaks(NamedTuple):
#     peak: int
#     amplitude: float
#     raise_time: float
#     decay_time: float
#     ... add whatever other data we need


# data = {
#     "A1": {  # well A1
#         "0000": {   # fov 0000
#             "0": ROIData(trace=[list[float]], peaks=[Peaks], mean_frequency=float, mean_amplitude=float, ...),  # roi 0  # noqa
#             "1": ROIData(trace=[list[float]], peaks=[Peaks], mean_frequency=float, mean_amplitude=float, ...),  # roi 1  # noqa
#             "2": ROIData(trace=[list[float]], peaks=[Peaks], mean_frequency=float, mean_amplitude=float, ...),  # roi 2  # noqa
#             },
#         "0001": {   # fov 0001
#             "0": ROIData(trace=[list[float]], peaks=[Peaks], mean_frequency=float, mean_amplitude=float, ...),  # roi 0 # noqa
#             "1": ROIData(trace=[list[float]], peaks=[Peaks], mean_frequency=float, mean_amplitude=float, ...),  # roi 1 # noqa
#             "2": ROIData(trace=[list[float]], peaks=[Peaks], mean_frequency=float, mean_amplitude=float, ...),  # roi 2 # noqa
#             },
#             ...  # more fovs
#         },
#     "A2": {  # well A2
#         "0000": {   # fov 0000
#             "0": ROIData(trace=[list[float]], peaks=[Peaks], mean_frequency=float, mean_amplitude=float, ...),  # roi 0 # noqa
#             "1": ROIData(trace=[list[float]], peaks=[Peaks], mean_frequency=float, mean_amplitude=float, ...),  # roi 1 # noqa
#             "2": ROIData(trace=[list[float]], peaks=[Peaks], mean_frequency=float, mean_amplitude=float, ...),  # roi 2 # noqa
#             },
#         "0001": {   # fov 0001
#             "0": ROIData(trace=[list[float]], peaks=[Peaks], mean_frequency=float, mean_amplitude=float, ...),  # roi 0 # noqa
#             "1": ROIData(trace=[list[float]], peaks=[Peaks], mean_frequency=float, mean_amplitude=float, ...),  # roi 1 # noqa
#             "2": ROIData(trace=[list[float]], peaks=[Peaks], mean_frequency=float, mean_amplitude=float, ...),  # roi 2 # noqa
#             },
#             ...  # more fovs
#         },
#     ...  # more wells
#     }

# the fov (or position) name we get from the metadata is:
#       "A1_0000", "A1_0001", "A1_0002", "A1_0003", ...

# therefore, we can split the name by "_" to get the well and fov names:
#       well, fov = name.split("_")

# once the 'data' dictionary is filled, we can access the data for a specific
# well, fov, and roi with:

#       data[well] -> will return the dictionary for that well containing all the fovs
#                     (e.g. ['0000', '0001', '0002', ...]).

#       data[well][fov] -> will return the dictionary for that fov containing all the
#                          rois indexes (e.g. ['0', '1', '2', ...]).

#       data[well][fov][roi] -> will return the ROIData object for that roi.
#           data[well][fov][roi].trace -> will return the trace data for that roi.
#           data[well][fov][roi].mean_frequency -> will return the mean frequency for that roi.  # noqa
#           data[well][fov][roi].mean_amplitude -> will return the mean amplitude for that roi.  # noqa
#           data[well][fov][roi].peaks -> will return a list of Peaks objects for that roi.  # noqa
#               data[well][fov][roi].peaks[0].peak -> will return the peak timepoint (frame) for the first peak  # noqa
#               data[well][fov][roi].peaks[0].amplitude -> will return the amplitude for the first peak  # noqa
#               data[well][fov][roi].peaks[0].raise_time -> will return the raise time for the first peak  # noqa
#               data[well][fov][roi].peaks[0].decay_time -> will return the decay time for the first peak  # noqa
#               ...


# we can also convert it to a MultiIndex DataFrame
# df = pd.concat(
#     {
#         (well_id, fov_id, roi_id): pd.Series(roi_data._asdict())
#         for well_id, fovs in data.items()
#         for fov_id, rois in fovs.items()
#         for roi_id, roi_data in rois.items()
#     },
#     names=["Well", "FOV", "ROI"],
# )

# # save to csv
# df.to_csv("output.csv")

# read from csv
# index_col=[0, 1, 2] specifies that the first three columns in the CSV file should be
# used as the MultiIndex, and header=0 specifies that the first row in the CSV file
# should be used as the header
# df = pd.read_csv('output.csv', index_col=[0, 1, 2], header=0)
