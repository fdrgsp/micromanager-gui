import useq

from micromanager_gui._plate_viewer._plate_map import PlateMapWidget

plate = useq.WellPlate.from_str("96-well")
p = PlateMapWidget()
p.setPlate(plate)
p.show()
