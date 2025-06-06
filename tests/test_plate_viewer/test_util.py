from micromanager_gui._plate_viewer._util import ROIData


class TestUtil:
    def test_replace(self):
        roi_instance = ROIData(well_fov_position="A1", dff=[1.0, 2.0, 3.0])
        updated_instance = roi_instance.replace(dff=[4.0, 5.0, 6.0])
        assert updated_instance.dff == [4.0, 5.0, 6.0]
