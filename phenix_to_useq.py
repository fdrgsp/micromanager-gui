from __future__ import annotations

import datetime as _dt
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import useq


# ────────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────────
def phenix_xml_to_mda_sequence(xml_path: str | Path) -> useq.MDASequence | None:
    """Convert Perkin-Elmer Phenix *Index.xml* → useq-schema MDASequence."""
    xml_path = Path(xml_path)
    root, ns = _get_root_and_ns(xml_path)

    # ── Stage positions (plate plan) ───────────────────────────────────────────
    plate_plan = _phenix_xml_to_plateplan(root, ns)
    if plate_plan is None:
        return None

    # ── Parse per-image metadata once; keyed by (row, col) ─────────────────────
    image_meta = _collect_image_metadata(root, ns)

    # ── Time plan (absolute times → interval loops) ────────────────────────────
    time_plan = _infer_time_plan(image_meta)

    # ── Channel plan (name/exposure pulled from <Maps>) ────────────────────────
    channel_plan = _parse_channel_map(root, ns)

    return useq.MDASequence(
        stage_positions=plate_plan,
        time_plan=time_plan,
        channels=channel_plan,
    )


# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
def _get_root_and_ns(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    ns = {"h": root.tag.split("}")[0].strip("{")}
    return root, ns


def _phenix_xml_to_plateplan(root: ET.Element, ns: dict[str, str]):
    plate = root.find(".//h:Plate", ns)
    if plate is None:
        return None

    rows = int(plate.findtext("h:PlateRows", namespaces=ns))
    cols = int(plate.findtext("h:PlateColumns", namespaces=ns))
    well_count_key = f"{rows * cols}-well"

    try:
        wellplate = useq.WellPlate.from_str(well_count_key)
    except ValueError:
        print(f"Plate {well_count_key} not in registry — add it if needed.")
        return None

    sel_rows, sel_cols = [], []
    for w in root.findall(".//h:Wells/h:Well", ns):
        sel_rows.append(int(w.findtext("h:Row", namespaces=ns)) - 1)
        sel_cols.append(int(w.findtext("h:Col", namespaces=ns)) - 1)

    return useq.WellPlatePlan(
        plate=wellplate,
        a1_center_xy=(0.0, 0.0),
        selected_wells=(tuple(sel_rows), tuple(sel_cols)),
    )


def _collect_image_metadata(root: ET.Element, ns: dict[str, str]):
    meta: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for img in root.findall(".//h:Images/h:Image", ns):
        md = {el.tag.split("}")[-1]: el.text for el in img}
        if md.get("Row") and md.get("Col"):
            meta[(int(md["Row"]), int(md["Col"]))].append(md)
    return meta


def _infer_time_plan(
    meta: dict[tuple[int, int], list[dict[str, str]]],
) -> useq.TIntervalLoops | None:
    """Return TIntervalLoops with interval=0 ms, loops = # timepoints."""
    # gather the distinct timepoint IDs present in the file
    t_ids = {
        int(m["TimepointID"])
        for frames in meta.values()
        for m in frames
        if m.get("TimepointID") is not None
    }

    if len(t_ids) < 2:  # single timepoint → no explicit time axis
        return None

    return useq.TIntervalLoops(
        loops=len(t_ids),
        interval=_dt.timedelta(milliseconds=0),  # fast-timelapse semantics
    )


def _parse_channel_map(root: ET.Element, ns: dict[str, str]) -> list[useq.Channel]:
    """Return one unique useq.Channel per ChannelID, exposure in ms."""
    info: dict[int, dict[str, float | str]] = {}

    for entry in root.findall(".//h:Maps/h:Map/h:Entry", ns):
        cid = int(entry.get("ChannelID", -1))

        # grab what is available in this <Entry>
        name = entry.findtext("h:ChannelName", default="", namespaces=ns).strip()
        exp_elem = entry.find("h:ExposureTime", ns)

        exposure_ms: float | None = None
        if exp_elem is not None and exp_elem.text:
            val = float(exp_elem.text)
            unit = (exp_elem.get("Unit", "s") or "s").lower()
            if unit.startswith("s"):  # seconds → ms
                exposure_ms = val * 1_000
            elif unit.startswith("us"):  # µs → ms
                exposure_ms = val / 1_000
            else:  # already in ms
                exposure_ms = val

        # update dict only if this entry adds new info
        rec = info.get(cid, {})
        if name:
            rec["name"] = name
        if exposure_ms is not None:
            rec["exp"] = exposure_ms
        info[cid] = rec

    # build Channel objects
    channels = [
        useq.Channel(
            config=rec.get("name", f"Ch{cid}"),
            exposure=rec.get("exp"),
        )
        for cid, rec in sorted(info.items())
    ]
    return channels


if __name__ == "__main__":
    from pymmcore_widgets.useq_widgets import WellPlateWidget
    from qtpy.QtWidgets import QApplication
    from rich import print

    app = QApplication([])
    seq = phenix_xml_to_mda_sequence("/Users/fdrgsp/Desktop/phenix/Index.xml")
    print(seq)
    if seq is not None and isinstance(seq.stage_positions, useq.WellPlatePlan):
        wdg = WellPlateWidget(plan=seq.stage_positions)
        wdg.show()
        app.exec()
