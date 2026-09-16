"""
LISA Traffic Lights dataset adapter.

The LISA Traffic Light Dataset (Jensen et al., UC San Diego) is a
day/night driving-sequence dataset for traffic-light state detection. The
original raw distribution ships as per-sequence video frame directories
plus CSV-format bounding-box annotations (not a single COCO JSON).

A pre-converted copy already in YOLO format (train/val/test + data.yaml)
is the common source here -- point ``configs/dataset/lisa.yaml`` straight
at its ``data.yaml`` and skip ``prepare_coco`` entirely. ``prepare_coco``
below targets the *original* raw CSV-annotated distribution, for when
starting from that instead.

Two annotation granularities exist upstream: "box" (one bounding box per
traffic light housing, this adapter's default) and "bulb" (finer boxes per
lit bulb region) -- both share the same 7-class taxonomy below. Pick
whichever your local conversion targets; this adapter assumes "box".

NOTE: the class list has been confirmed against a real converted "box"
export (7 classes, same names/order as both the "box" and "bulb" variants).

Stats: see docs/datasets/lisa/README.md (class distribution, split
summary, box geometry -- generated via detectionbench-dataset-stats).
"""

from __future__ import annotations

from pathlib import Path

from detectionbench.datasets.base import DatasetAdapter, DatasetSpec
from detectionbench.datasets.registry import register

_CLASSES = [
    "go",
    "goForward",
    "goLeft",
    "stop",
    "stopLeft",
    "warning",
    "warningLeft",
]


@register
class LISATrafficLightsAdapter(DatasetAdapter):
    """Adapter for the LISA Traffic Light Dataset (day/night driving sequences)."""

    spec = DatasetSpec(
        key="lisa",
        display_name="LISA Traffic Lights",
        classes=_CLASSES,
        description=(
            "The LISA Traffic Light Dataset is a benchmark for traffic-light detection "
            "and recognition: continuous day/night video sequences recorded in San "
            "Diego, California, under varying light and weather. It's used to "
            "benchmark autonomous-vehicle and ADAS perception systems on a safety- "
            "critical, small-object detection task; this adapter carries the box- "
            "annotation variant."
        ),
        homepage="https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset",
        citation=(
            "@article{jensen2016vision,\n"
            "  title={Vision for looking at traffic lights: Issues, survey, and perspectives},\n"  # noqa: E501
            "  author={Jensen, Morten Born{\\o} and Philipsen, Mark Philip and M{\\o}gelmose, Andreas and Moeslund, Thomas Baltzer and Trivedi, Mohan Manubhai},\n"  # noqa: E501
            "  journal={IEEE Transactions on Intelligent Transportation Systems},\n"
            "  volume={17},\n"
            "  number={7},\n"
            "  pages={1800--1815},\n"
            "  year={2016},\n"
            "  doi={10.1109/TITS.2015.2509509},\n"
            "  publisher={IEEE}\n"
            "}"
        ),
        license="CC BY-NC-SA 4.0 -- see homepage before redistributing.",
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert a raw LISA Traffic Lights download into canonical COCO splits."""
        raise NotImplementedError(
            "LISA adapter is not implemented yet for the original raw "
            "distribution (per-sequence frames + CSV annotations). A "
            "pre-converted YOLO copy bypasses this entirely -- see the "
            "module docstring."
        )
