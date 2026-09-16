"""
Brackish (Brackish Underwater) dataset adapter.

Brackish is an underwater marine-animal detection dataset collected via a
camera mounted 9m below the surface on the Limfjords bridge in northern
Denmark by Aalborg University. Same domain as the "Aquarium" dataset in
the original DetectionBench roadmap (see ``Objectives.md``) -- Brackish is
a candidate substitute/addition, not yet an official roadmap decision.

A Roboflow-exported copy already in YOLO format (train/valid/test +
data.yaml) is the common source -- point ``configs/dataset/brackish.yaml``
straight at its ``data.yaml`` and skip ``prepare_coco`` entirely.
``prepare_coco`` below targets the *original* raw distribution (extracted
video frames + annotations from Aalborg University), for when starting
from that instead.

NOTE: the class list below has been confirmed against a real Roboflow
"Brackish Underwater" export (6 classes, same names/order). ~15% of images
per split have zero annotations (background-only frames) -- unlike
ExDark, empty label files are expected and normal for this dataset.

Stats: see docs/datasets/brackish/README.md (class distribution, split
summary, box geometry -- generated via detectionbench-dataset-stats).
"""

from __future__ import annotations

from pathlib import Path

from detectionbench.datasets.base import DatasetAdapter, DatasetSpec
from detectionbench.datasets.registry import register

_CLASSES = [
    "crab",
    "fish",
    "jellyfish",
    "shrimp",
    "small_fish",
    "starfish",
]


@register
class BrackishAdapter(DatasetAdapter):
    """Adapter for the Brackish Underwater marine-animal dataset."""

    spec = DatasetSpec(
        key="brackish",
        display_name="Brackish Underwater",
        classes=_CLASSES,
        description=(
            "Brackish is an underwater object-detection benchmark of fish, crabs, and "
            "other marine animals, collected with a camera mounted 9 meters below the "
            "surface in Limfjorden, a brackish strait near Aalborg, Denmark, under "
            "naturally varying visibility conditions. It's used to benchmark "
            "underwater detection where color cast, turbidity, and particulate matter "
            "degrade standard detectors."
        ),
        homepage="https://www.kaggle.com/datasets/aalborguniversity/brackish-dataset",
        citation=(
            "@InProceedings{pedersen2019brackish,\n"
            "  title = {Detection of Marine Animals in a New Underwater Dataset with Varying Visibility},\n"  # noqa: E501
            "  author = {Pedersen, Malte and Haurum, Joakim Bruslund and Gade, Rikke and Moeslund, Thomas B. and Madsen, Niels},\n"  # noqa: E501
            "  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},\n"  # noqa: E501
            "  month = {June},\n"
            "  year = {2019}\n"
            "}"
        ),
        license="CC BY 4.0 -- see homepage before redistributing.",
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert a raw Brackish download into canonical COCO splits."""
        raise NotImplementedError(
            "Brackish adapter is not implemented yet for the original raw "
            "distribution. A Roboflow-exported YOLO copy bypasses this "
            "entirely -- see the module docstring."
        )
