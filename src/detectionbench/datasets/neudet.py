"""
NEU-DET (hot-rolled steel strip surface defect detection) adapter.

Source: the official release from Kechen Song's group at Northeastern
University -- a flat, undivided ``NEU-DET/IMAGES/*.jpg`` (1,800 200x200
grayscale images, 300 per class) plus one Pascal VOC XML per image under
``NEU-DET/ANNOTATIONS/`` (bounding boxes; class name in ``object/name``).

NEU-DET ships with **no** official train/val/test split -- this adapter
creates a deterministic seeded 80/10/10 split (``_SPLIT_SEED`` /
``_SPLIT_RATIOS``, matching the same approach used for GC10-DET, which has
the same no-official-split situation). Adjust those constants to match a
specific external split if you need one.

License: no license stated anywhere -- the official homepage
(faculty.neu.edu.cn/songkc/...) requests only a citation, no redistribution
grant. Fine to build/evaluate against locally, **not** to re-host. No
Hugging Face mirror without the maintainer's written permission -- see
``detectionbench-download-dataset --dataset neudet`` for the official
download locations instead.

Stats: see docs/datasets/neudet/README.md (class distribution, split
summary, box geometry -- generated via detectionbench-dataset-stats).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any
from xml.etree import ElementTree  # noqa: S405  # nosec: B405

from detectionbench.datasets.base import (
    COCO_ANNOTATION_FILENAME,
    DatasetAdapter,
    DatasetSpec,
    link_image,
)
from detectionbench.datasets.registry import register

_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
_CLASS_INDEX = {name: index for index, name in enumerate(_CLASSES)}

# NEU-DET has no upstream split -- this adapter creates one deterministically
# (same approach as GC10-DET, which is in the same no-official-split situation).
_SPLIT_SEED = 42
_SPLIT_RATIOS = (0.8, 0.1)  # (train, valid); test gets the remainder.


@register
class NEUDETAdapter(DatasetAdapter):
    """Adapter for the NEU-DET steel-surface-defect detection dataset."""

    spec = DatasetSpec(
        key="neudet",
        display_name="NEU-DET",
        classes=_CLASSES,
        description=(
            "NEU-DET is a hot-rolled steel strip surface-defect detection benchmark: "
            "1,800 grayscale 200x200 images (300 per class) across six defect types -- "
            "crazing, inclusion, patches, pitted surface, rolled-in scale, and "
            "scratches. It's used to benchmark automated defect localization for "
            "steel-manufacturing quality control, complementing GC10-DET's "
            "metallic-surface-defect taxonomy with a different steel-inspection domain."
        ),
        homepage="http://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/index.htm",
        citation=(
            "@article{he2020end,\n"
            "  title={An End-to-end Steel Surface Defect Detection Approach via "
            "Fusing Multiple Hierarchical Features},\n"
            "  author={He, Yu and Song, Kechen and Meng, Qinggang and Yan, "
            "Yunhui},\n"
            "  journal={IEEE Transactions on Instrumentation and Measurement},\n"
            "  volume={69},\n"
            "  number={4},\n"
            "  pages={1493--1504},\n"
            "  year={2020}\n"
            "}"
        ),
        license="Unclear -- no explicit grant, citation-requested only. See homepage.",
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert the official NEU-DET release into the canonical COCO layout."""
        image_dir = raw_dir / "IMAGES"
        annotation_dir = raw_dir / "ANNOTATIONS"
        if not image_dir.is_dir() or not annotation_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {raw_dir}/IMAGES and {raw_dir}/ANNOTATIONS."
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(image_dir.glob("*.jpg"))
        shuffled = image_paths[:]
        random.Random(_SPLIT_SEED).shuffle(shuffled)  # noqa: S311  # nosec: B311

        n_total = len(shuffled)
        n_train = int(n_total * _SPLIT_RATIOS[0])
        n_valid = int(n_total * _SPLIT_RATIOS[1])
        members = {
            "train": shuffled[:n_train],
            "valid": shuffled[n_train : n_train + n_valid],
            "test": shuffled[n_train + n_valid :],
        }
        for split_name, split_images in members.items():
            _convert_split(split_images, annotation_dir, output_dir / split_name)


def _convert_split(
    image_paths: list[Path], annotation_dir: Path, split_output_dir: Path
) -> None:
    """Parse one split's Pascal VOC XML annotations into a canonical COCO JSON."""
    split_output_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1

    for image_id, image_path in enumerate(image_paths, start=1):
        xml_path = annotation_dir / f"{image_path.stem}.xml"
        if not xml_path.exists():
            continue
        # Local, trusted dataset annotation files (not untrusted network input).
        root = ElementTree.parse(xml_path).getroot()  # noqa: S314  # nosec: B314
        size = root.find("size")
        if size is None:
            continue
        width = int(size.findtext("width", "0"))
        height = int(size.findtext("height", "0"))

        link_image(image_path, split_output_dir / image_path.name)
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        for obj in root.findall("object"):
            class_id = _CLASS_INDEX.get((obj.findtext("name") or "").strip())
            if class_id is None:
                continue
            box = obj.find("bndbox")
            if box is None:
                continue
            x_min = float(box.findtext("xmin", "0"))
            y_min = float(box.findtext("ymin", "0"))
            x_max = float(box.findtext("xmax", "0"))
            y_max = float(box.findtext("ymax", "0"))
            box_width = x_max - x_min
            box_height = y_max - y_min
            if box_width <= 0 or box_height <= 0:
                continue
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [x_min, y_min, box_width, box_height],
                    "area": box_width * box_height,
                    "segmentation": [],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    payload = {
        "info": {"description": f"NEU-DET canonical COCO ({split_output_dir.name})"},
        "licenses": [{"id": 1, "name": "See dataset homepage", "url": ""}],
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
