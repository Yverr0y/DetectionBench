"""
HRSID (High-Resolution SAR Images Dataset) adapter -- ship detection.

Source: the original ``HRSID_JPG`` release -- ``JPEGImages/*.jpg`` (5,604
800x800 SAR crops) plus COCO-format ``annotations/{train2017,test2017}.json``
(one ``ship`` category, id 1; boxes as COCO ``[x, y, w, h]``; polygon
segmentation is present but unused for detection).

HRSID's own split (``train2017`` 3,642 / ``test2017`` 1,962) is a random
split over image crops, not over parent scenes -- so this adapter keeps
``test2017`` as ``test`` and carves a validation set out of ``train2017``
by a seeded random crop split (``_VAL_FRACTION``), consistent with how
upstream split train vs. test.

License: unclear for the data. The GitHub repo (chaozhong2010/HRSID)
carries a GPL-3.0 ``LICENSE`` (a software licence) and asks only for
citation; the imagery is partly TerraSAR-X / TanDEM-X (DLR, scientific-use,
not freely redistributable). Fine to build/evaluate against locally; **no
Hugging Face mirror** without the authors confirming redistribution terms.

Stats: see docs/datasets/hrsid/README.md (class distribution, split
summary, box geometry -- generated via detectionbench-dataset-stats).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from detectionbench.datasets.base import (
    COCO_ANNOTATION_FILENAME,
    DatasetAdapter,
    DatasetSpec,
    link_image,
)
from detectionbench.datasets.registry import register

_CLASSES = ["ship"]

# HRSID annotation file (stem) -> canonical split(s). ``train2017`` is split.
_TEST_ANNOTATION = "test2017"
_TRAIN_ANNOTATION = "train2017"
_VAL_FRACTION = 0.15
_SPLIT_SEED = 42


@register
class HRSIDAdapter(DatasetAdapter):
    """Adapter for the HRSID high-resolution SAR ship-detection dataset."""

    spec = DatasetSpec(
        key="hrsid",
        display_name="HRSID",
        classes=_CLASSES,
        description=(
            "HRSID is a high-resolution Synthetic Aperture Radar (SAR) benchmark for "
            "ship detection: 5,604 800x800 image crops (from 136 larger scenes) with "
            "16,951 ship instances, spanning multiple resolutions, polarizations, sea "
            "states, and coastal/open-sea conditions, sourced from Sentinel-1B, "
            "TerraSAR-X, and TanDEM-X. It's used to benchmark ship detection in SAR "
            "imagery, where speckle noise and side-lobe artifacts make optical-trained "
            "detectors unreliable."
        ),
        homepage="https://github.com/chaozhong2010/HRSID",
        citation=(
            "@ARTICLE{wei2020hrsid,\n"
            "  author={Wei, Shunjun and Zeng, Xiangfeng and Qu, Qizhe and Wang, Mou and Su, Hao and Shi, Jun},\n"  # noqa: E501
            "  journal={IEEE Access},\n"
            "  title={HRSID: A High-Resolution SAR Images Dataset for Ship Detection and Instance Segmentation},\n"  # noqa: E501
            "  year={2020},\n"
            "  volume={8},\n"
            "  pages={120234-120254},\n"
            "  doi={10.1109/ACCESS.2020.3005861}\n"
            "}"
        ),
        license="Unclear (repo LICENSE is GPL-3.0); imagery partly DLR-restricted.",
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert the HRSID_JPG release into the canonical COCO layout."""
        image_dir = raw_dir / "JPEGImages"
        annotation_dir = raw_dir / "annotations"
        if not image_dir.is_dir() or not annotation_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {raw_dir}/JPEGImages and {raw_dir}/annotations."
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        train_data = json.loads(
            (annotation_dir / f"{_TRAIN_ANNOTATION}.json").read_text(encoding="utf-8")
        )
        train_images = sorted(train_data["images"], key=lambda im: im["file_name"])
        shuffled = train_images[:]
        random.Random(_SPLIT_SEED).shuffle(shuffled)  # noqa: S311  # nosec: B311
        n_val = max(1, round(len(shuffled) * _VAL_FRACTION))
        val_ids = {im["id"] for im in shuffled[:n_val]}

        _write_split(
            [im for im in train_images if im["id"] not in val_ids],
            train_data["annotations"],
            image_dir,
            output_dir / "train",
        )
        _write_split(
            [im for im in train_images if im["id"] in val_ids],
            train_data["annotations"],
            image_dir,
            output_dir / "valid",
        )
        test_data = json.loads(
            (annotation_dir / f"{_TEST_ANNOTATION}.json").read_text(encoding="utf-8")
        )
        _write_split(
            test_data["images"],
            test_data["annotations"],
            image_dir,
            output_dir / "test",
        )


def _write_split(
    split_images: list[dict[str, Any]],
    all_annotations: list[dict[str, Any]],
    image_dir: Path,
    split_output_dir: Path,
) -> None:
    """Emit one canonical COCO split from a subset of a HRSID COCO file."""
    split_output_dir.mkdir(parents=True, exist_ok=True)

    keep_image_ids = {im["id"] for im in split_images}
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for ann in all_annotations:
        if ann["image_id"] not in keep_image_ids:
            continue
        x, y, box_width, box_height = ann["bbox"]
        if box_width <= 0 or box_height <= 0:
            continue
        annotations.append(
            {
                "id": annotation_id,
                "image_id": ann["image_id"],
                "category_id": 0,  # single class; remap HRSID's id 1 -> 0
                "bbox": [x, y, box_width, box_height],
                "area": float(box_width) * float(box_height),
                "segmentation": [],
                "iscrowd": int(ann.get("iscrowd", 0)),
            }
        )
        annotation_id += 1

    images: list[dict[str, Any]] = []
    for im in split_images:
        src = image_dir / im["file_name"]
        if src.exists():
            link_image(src, split_output_dir / im["file_name"])
        images.append(
            {
                "id": im["id"],
                "file_name": im["file_name"],
                "width": im["width"],
                "height": im["height"],
            }
        )

    payload = {
        "info": {"description": f"HRSID canonical COCO ({split_output_dir.name})"},
        "licenses": [
            {"id": 1, "name": "See dataset homepage", "url": HRSIDAdapter.spec.homepage}
        ],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": _CLASSES[0], "supercategory": "none"}],
    }
    (split_output_dir / COCO_ANNOTATION_FILENAME).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"[{split_output_dir.name}] {len(images)} images, {len(annotations)} boxes")
