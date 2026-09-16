"""
ExDark (Exclusively Dark Image Dataset) adapter.

ExDark is a low-light robustness benchmark. It was originally released as
an image-classification dataset organized into one subdirectory per class
(``ExDark/Bicycle/*.jpg``, ``ExDark/Boat/*.jpg``, ...); bounding-box
annotations were added later by the community, typically as one text file
per image (Pascal-VOC- or YOLO-style, not a single COCO JSON) -- see
https://github.com/cs-chan/Exclusively-Dark-Image-Dataset.

In practice, a Roboflow-exported copy already in YOLO format (train/valid/test
+ data.yaml) is a common, much simpler source -- point
``configs/dataset/exdark.yaml`` straight at its ``data.yaml`` and skip
``prepare_coco`` entirely. ``prepare_coco`` below targets the *original* raw
per-class-folder distribution, for when starting from that instead: it will
need to walk the per-class directory tree, parse each image's bounding-box
annotation file, and emit the canonical
``{train,valid,test}/_annotations.coco.json`` layout.

NOTE: the class list below has been confirmed against a real Roboflow
"Exclusively-Dark-Image" export (12 classes, same names/order).

Stats: see docs/datasets/exdark/README.md (class distribution, split
summary, box geometry -- generated via detectionbench-dataset-stats).
"""

from __future__ import annotations

from pathlib import Path

from detectionbench.datasets.base import DatasetAdapter, DatasetSpec
from detectionbench.datasets.registry import register

_CLASSES = [
    "Bicycle",
    "Boat",
    "Bottle",
    "Bus",
    "Car",
    "Cat",
    "Chair",
    "Cup",
    "Dog",
    "Motorbike",
    "People",
    "Table",
]


@register
class ExDarkAdapter(DatasetAdapter):
    """Adapter for the Exclusively Dark Image Dataset (low-light robustness)."""

    spec = DatasetSpec(
        key="exdark",
        display_name="ExDark",
        classes=_CLASSES,
        description=(
            "ExDark (Exclusively Dark Image Dataset) is a low-light robustness "
            "benchmark: 7,344 images captured across 10 low-light conditions, from "
            "very low light to twilight, with both image-level class labels and "
            "object-level bounding boxes across 12 classes. It's used to study object "
            "detection robustness under degraded illumination, a regime standard COCO- "
            "trained detectors handle poorly."
        ),
        homepage="https://github.com/cs-chan/Exclusively-Dark-Image-Dataset",
        citation=(
            "@article{Exdark,\n"
            "  title = {Getting to Know Low-light Images with The Exclusively Dark Dataset},\n"  # noqa: E501
            "  author = {Loh, Yuen Peng and Chan, Chee Seng},\n"
            "  journal = {Computer Vision and Image Understanding},\n"
            "  volume = {178},\n"
            "  pages = {30-42},\n"
            "  year = {2019},\n"
            "  doi = {https://doi.org/10.1016/j.cviu.2018.10.010}\n"
            "}"
        ),
        license="BSD-3-Clause -- see homepage before redistributing.",
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert a raw ExDark download into canonical COCO splits."""
        raise NotImplementedError(
            "ExDark adapter is not implemented yet. Expected raw format: "
            "per-class image subdirectories plus per-image bounding-box "
            "annotation files (not a single COCO JSON). Implement this once "
            "a raw copy of the dataset is available."
        )
