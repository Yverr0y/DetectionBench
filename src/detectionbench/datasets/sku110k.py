"""
SKU-110K (dense retail-shelf object detection) adapter.

Source: the official Trax/eg4000 release (``SKU110K_fixed.tar.gz``) --
``images/*.jpg`` plus CSV annotations, one row per box, no header:

    image_name,x1,y1,x2,y2,class,image_width,image_height

split across ``annotations/annotations_{train,val,test}.csv``. All boxes
share a single class, literally the string ``object`` -- the "110K" in the
name refers to the number of distinct SKUs *pictured*, not detection
classes; the task is single-class, extremely dense object detection
(shelf items packed edge to edge). The three CSVs reference filenames in
one shared ``images/`` directory (not per-split subdirectories).

License: distributed "for the exclusive use by the recipient and solely
for academic and non-commercial purposes" -- more restrictive than a
generic non-commercial clause (no redistribution to third parties implied
at all). Fine to build/evaluate against locally, **not** to re-host. No
Hugging Face mirror without the authors' written permission -- see
``detectionbench-download-dataset --dataset sku110k`` (a direct S3 URL, no
Google Drive/Baidu dance needed for this one).

Stats: see docs/datasets/sku110k/README.md (class distribution, split
summary, box geometry -- generated via detectionbench-dataset-stats).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image

from detectionbench.datasets.base import (
    COCO_ANNOTATION_FILENAME,
    DatasetAdapter,
    DatasetSpec,
    link_image,
)
from detectionbench.datasets.registry import register

_CLASSES = ["object"]
_CSV_FIELDS = (
    "file_name",
    "x1",
    "y1",
    "x2",
    "y2",
    "class",
    "image_width",
    "image_height",
)

# annotations_<source_split>.csv -> canonical split name.
_SPLIT_FILES = {"train": "train", "val": "valid", "test": "test"}


@register
class SKU110KAdapter(DatasetAdapter):
    """Adapter for the SKU-110K dense retail-shelf detection dataset."""

    spec = DatasetSpec(
        key="sku110k",
        display_name="SKU-110K",
        classes=_CLASSES,
        description=(
            "SKU-110K is a dense, single-class retail-shelf object-detection "
            "benchmark: 11,743 images of store shelves with items packed edge to edge. "
            "It's used to stress-test detectors on extreme object density and overlap "
            "rather than fine-grained SKU classification -- despite the name, every "
            "box is labeled a single class, `object`."
        ),
        homepage="https://github.com/eg4000/SKU110K_CVPR19",
        citation=(
            "@inproceedings{goldman2019dense,\n"
            "  title={Precise Detection in Densely Packed Scenes},\n"
            "  author={Goldman, Eran and Herzig, Roei and Eisenschtat, Aviv and Goldberger, Jacob and Hassner, Tal},\n"  # noqa: E501
            "  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},\n"  # noqa: E501
            "  pages={5227--5236},\n"
            "  year={2019}\n"
            "}"
        ),
        license=(
            "Exclusive use by the recipient, academic/non-commercial only "
            "-- no redistribution. See homepage."
        ),
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert the official SKU-110K release into the canonical COCO layout."""
        image_dir = raw_dir / "images"
        annotation_dir = raw_dir / "annotations"
        if not image_dir.is_dir() or not annotation_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {raw_dir}/images and {raw_dir}/annotations."
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        for source_split, target_split in _SPLIT_FILES.items():
            csv_path = annotation_dir / f"annotations_{source_split}.csv"
            if not csv_path.exists():
                continue
            _convert_split(csv_path, image_dir, output_dir / target_split)


def _convert_split(csv_path: Path, image_dir: Path, split_output_dir: Path) -> None:
    """Parse one split's headerless CSV annotations into a canonical COCO JSON."""
    split_output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_image: dict[str, list[dict[str, str]]] = {}
    with csv_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file, fieldnames=_CSV_FIELDS):
            rows_by_image.setdefault(row["file_name"], []).append(row)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1

    for image_id, (file_name, rows) in enumerate(
        sorted(rows_by_image.items()), start=1
    ):
        src = image_dir / file_name
        if not src.exists():
            continue
        width, height = _image_size(src, rows[0])
        link_image(src, split_output_dir / file_name)
        images.append(
            {"id": image_id, "file_name": file_name, "width": width, "height": height}
        )

        for row in rows:
            try:
                x1, y1, x2, y2 = (float(row[k]) for k in ("x1", "y1", "x2", "y2"))
            except (TypeError, ValueError):
                continue
            box_width = x2 - x1
            box_height = y2 - y1
            if box_width <= 0 or box_height <= 0:
                continue
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 0,
                    "bbox": [x1, y1, box_width, box_height],
                    "area": box_width * box_height,
                    "segmentation": [],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    payload = {
        "info": {"description": f"SKU-110K canonical COCO ({split_output_dir.name})"},
        "licenses": [{"id": 1, "name": "See dataset homepage", "url": ""}],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": "object", "supercategory": "none"}],
    }
    (split_output_dir / COCO_ANNOTATION_FILENAME).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"[{split_output_dir.name}] {len(images)} images, {len(annotations)} boxes")


def _image_size(image_path: Path, sample_row: dict[str, str]) -> tuple[int, int]:
    """Read width/height from the CSV row, falling back to the image file itself."""
    try:
        return int(sample_row["image_width"]), int(sample_row["image_height"])
    except (KeyError, TypeError, ValueError):
        with Image.open(image_path) as image:
            return image.size
