"""
DUO (Detecting Underwater Objects) adapter -- underwater robot-picking detection.

Source: the official DUO release -- ``images/{train,test}/*.jpg`` plus
COCO-format ``annotations/instances_{train,test}.json`` (4 categories, ids
1-4: holothurian, echinus, scallop, starfish).

DUO re-annotates and merges URPC2017-2020 + UDD to fix annotation-quality
issues in those source datasets; its own split is train (6,671 images) /
test (1,111 images), with no validation set. This adapter keeps ``test`` as
``test`` and carves a seeded random validation slice out of ``train``
(``_VAL_FRACTION``), the same approach used for the HRSID adapter (DUO's
own train/test split is likewise a random split, not scene-aware).

License: no license is stated by the DUO authors (GitHub, paper) or by the
URPC contest data it re-annotates -- URPC access historically required
signing a data-use commitment letter, and post-contest download links were
withdrawn. Fine to build/evaluate against locally, **not** to re-host. No
Hugging Face mirror without the authors' written permission -- see
``detectionbench-download-dataset --dataset duo`` for the official download
locations instead.
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

_CLASSES = ["holothurian", "echinus", "scallop", "starfish"]
_SOURCE_CATEGORY_ID_OFFSET = 1  # DUO's category ids are 1-4; canonical wants 0-3.

_VAL_FRACTION = 0.15
_SPLIT_SEED = 42


@register
class DUOAdapter(DatasetAdapter):
    """Adapter for the DUO underwater robot-picking object-detection dataset."""

    spec = DatasetSpec(
        key="duo",
        display_name="DUO",
        classes=_CLASSES,
        homepage="https://github.com/chongweiliu/DUO",
        citation=(
            "Liu et al., 'A Dataset and Benchmark of Underwater Object "
            "Detection for Robot Picking', ICME Workshops 2021 "
            "(arXiv:2106.05681)."
        ),
        license="Unclear -- no explicit grant; re-annotates gated URPC contest data.",
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert the official DUO release into the canonical COCO layout."""
        image_dir = raw_dir / "images"
        annotation_dir = raw_dir / "annotations"
        if not image_dir.is_dir() or not annotation_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {raw_dir}/images and {raw_dir}/annotations."
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        train_data = json.loads(
            (annotation_dir / "instances_train.json").read_text(encoding="utf-8")
        )
        train_images = sorted(train_data["images"], key=lambda im: im["file_name"])
        shuffled = train_images[:]
        random.Random(_SPLIT_SEED).shuffle(shuffled)  # noqa: S311  # nosec: B311
        n_val = max(1, round(len(shuffled) * _VAL_FRACTION))
        val_ids = {im["id"] for im in shuffled[:n_val]}

        _write_split(
            [im for im in train_images if im["id"] not in val_ids],
            train_data["annotations"],
            image_dir / "train",
            output_dir / "train",
        )
        _write_split(
            [im for im in train_images if im["id"] in val_ids],
            train_data["annotations"],
            image_dir / "train",
            output_dir / "valid",
        )
        test_data = json.loads(
            (annotation_dir / "instances_test.json").read_text(encoding="utf-8")
        )
        _write_split(
            test_data["images"],
            test_data["annotations"],
            image_dir / "test",
            output_dir / "test",
        )


def _write_split(
    split_images: list[dict[str, Any]],
    all_annotations: list[dict[str, Any]],
    image_dir: Path,
    split_output_dir: Path,
) -> None:
    """Emit one canonical COCO split from a subset of a DUO COCO file."""
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
                "category_id": ann["category_id"] - _SOURCE_CATEGORY_ID_OFFSET,
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
        "info": {"description": f"DUO canonical COCO ({split_output_dir.name})"},
        "licenses": [
            {"id": 1, "name": "See dataset homepage", "url": DUOAdapter.spec.homepage}
        ],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index, "name": name, "supercategory": "none"}
            for index, name in enumerate(_CLASSES)
        ],
    }
    (split_output_dir / COCO_ANNOTATION_FILENAME).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"[{split_output_dir.name}] {len(images)} images, {len(annotations)} boxes")
