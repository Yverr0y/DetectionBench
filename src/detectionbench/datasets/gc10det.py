"""
GC10-DET (metallic surface defect detection) adapter.

Source: the Dataset Ninja export of GC10-DET, which ships as a single
undivided ``ds/`` folder -- ``ds/img/*.jpg`` plus one Supervisely-format
``ds/ann/<image>.jpg.json`` per image. Each annotation holds the image
size under ``size`` and axis-aligned boxes under
``objects[].points.exterior`` as ``[[x1, y1], [x2, y2]]`` (only
``geometryType == "rectangle"`` objects are used).

The GC10-DET paper (Lv et al., *Sensors* 2020, 20(6):1562) defines **no**
official train/val/test split, so this adapter creates a deterministic
seeded split (``_SPLIT_SEED`` / ``_SPLIT_RATIOS``, currently 80/10/10 over
the sorted-then-shuffled image list). Adjust those constants to match a
specific external split if you need one.

License: CC BY 4.0 -- attribution required; redistribution and derivative
datasets are permitted. The Supervisely class title ``"waist folding"`` is
normalized to ``waist_folding`` for consistency with the other nine.
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

_CLASSES = [
    "crease",
    "crescent_gap",
    "inclusion",
    "oil_spot",
    "punching_hole",
    "rolled_pit",
    "silk_spot",
    "waist_folding",
    "water_spot",
    "welding_line",
]
_CLASS_INDEX = {name: index for index, name in enumerate(_CLASSES)}

_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}
_RECTANGLE = "rectangle"
_BOX_CORNER_COUNT = 2

# GC10-DET has no upstream split -- this adapter creates one deterministically.
_SPLIT_SEED = 42
_SPLIT_RATIOS = (0.8, 0.1)  # (train, valid); test gets the remainder.


def _normalize_title(title: str) -> str:
    """Map a Supervisely class title to its canonical snake_case class name."""
    return title.strip().replace(" ", "_")


@register
class GC10DetAdapter(DatasetAdapter):
    """Adapter for the GC10-DET metallic-surface-defect detection dataset."""

    spec = DatasetSpec(
        key="gc10det",
        display_name="GC10-DET",
        classes=_CLASSES,
        homepage="https://github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets",
        citation=(
            "Lv, Duan, Jiang, Fu, Gan, 'Deep Metallic Surface Defect "
            "Detection: The New Benchmark and Detection Network', Sensors, "
            "20(6):1562, 2020."
        ),
        license="CC BY 4.0 -- attribution required; redistribution permitted.",
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert the GC10-DET Dataset Ninja export into the canonical COCO layout."""
        image_dir = raw_dir / "ds" / "img"
        annotation_dir = raw_dir / "ds" / "ann"
        if not image_dir.is_dir() or not annotation_dir.is_dir():
            raise FileNotFoundError(
                f"Expected a Dataset Ninja layout at {raw_dir}/ds/(img|ann); not found."
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in _IMAGE_EXTENSIONS
        )
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
    """Parse one split's Supervisely annotations into a canonical COCO JSON."""
    split_output_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1

    for image_id, image_path in enumerate(image_paths, start=1):
        annotation_path = annotation_dir / f"{image_path.name}.json"
        if not annotation_path.exists():
            continue
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        width = int(data["size"]["width"])
        height = int(data["size"]["height"])

        link_image(image_path, split_output_dir / image_path.name)
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        for obj in data.get("objects", []):
            if obj.get("geometryType") != _RECTANGLE:
                continue
            class_id = _CLASS_INDEX.get(_normalize_title(obj["classTitle"]))
            if class_id is None:
                continue
            exterior = obj["points"]["exterior"]
            if len(exterior) != _BOX_CORNER_COUNT:
                continue
            (x1, y1), (x2, y2) = exterior
            x_min, x_max = sorted((float(x1), float(x2)))
            y_min, y_max = sorted((float(y1), float(y2)))
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
        "info": {"description": f"GC10-DET canonical COCO ({split_output_dir.name})"},
        "licenses": [
            {
                "id": 1,
                "name": "CC BY 4.0",
                "url": "https://creativecommons.org/licenses/by/4.0/",
            }
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
