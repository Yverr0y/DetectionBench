"""
RDD2022 (multi-national Road Damage Detection 2022) adapter.

Expects the raw input already in a split-first Ultralytics-style YOLO tree
(the layout of the ``RDD_SPLIT`` export)::

    raw_dir/
    ├── train/{images,labels}/
    ├── val/{images,labels}/
    └── test/{images,labels}/

Labels are normalized ``class xc yc w h`` text files, one per image. This
adapter converts that straight into the canonical COCO layout.

**Class taxonomy.** The CRDDC2022 challenge scores four damage types --
D00/D10/D20/D40 -> ``longitudinal_crack`` / ``transverse_crack`` /
``alligator_crack`` / ``pothole`` (ids 0-3 here). The source ``RDD_SPLIT``
labels also carry a 5th id (``4``, an "other" / D50-style bucket, ~6.5k
boxes); those are **dropped** during conversion to keep the 4-class
taxonomy the project standardizes on. The per-split count of dropped
boxes is printed as ``skipped invalid boxes``.

License: RDD2022 images are **CC BY-SA 4.0**. Any redistribution (a
Hugging Face mirror included) must stay under CC BY-SA 4.0 and attribute
the original authors (Arya et al., arXiv:2209.08538).

Stats: see docs/datasets/rdd2022/README.md (class distribution, split
summary, box geometry -- generated via detectionbench-dataset-stats).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from detectionbench.datasets.base import (
    COCO_ANNOTATION_FILENAME,
    DatasetAdapter,
    DatasetSpec,
    link_image,
)
from detectionbench.datasets.registry import register
from detectionbench.utils.convert_yolo_to_coco import (
    IMAGE_EXTENSIONS,
    LabelConversionContext,
    convert_yolo_label_line,
    load_image_size,
)

_CLASSES = [
    "longitudinal_crack",
    "transverse_crack",
    "alligator_crack",
    "pothole",
]

# RDD_SPLIT's split dir names -> canonical roboflow-style split names.
_SPLIT_MAP = {"train": "train", "val": "valid", "test": "test"}


@register
class RDD2022Adapter(DatasetAdapter):
    """Adapter for the multi-national Road Damage Detection 2022 dataset."""

    spec = DatasetSpec(
        key="rdd2022",
        display_name="RDD2022 Road Damage",
        classes=_CLASSES,
        description=(
            "RDD2022 is a multi-national street-level road-damage detection benchmark: "
            "47,420 road images from six countries (Japan, India, Czech Republic, "
            "Norway, United States, China), captured with vehicle-mounted smartphones, "
            "dashboard cameras, and drones, annotated for pavement distress across "
            "four CRDDC2022 damage types. It's used to benchmark automatic road- "
            "condition assessment across diverse road types, imaging setups, and "
            "damage conventions."
        ),
        homepage="https://github.com/sekilab/RoadDamageDetector",
        citation=(
            "@article{arya2022rdd2022,\n"
            "  title = {RDD2022: A multi-national image dataset for automatic Road Damage Detection},\n"  # noqa: E501
            "  author = {Arya, Deeksha and Maeda, Hiroya and Ghosh, Sanjay Kumar and Toshniwal, Durga and Sekimoto, Yoshihide},\n"  # noqa: E501
            "  journal = {arXiv preprint arXiv:2209.08538},\n"
            "  year = {2022}\n"
            "}"
        ),
        license="CC BY-SA 4.0 -- share-alike; keep any mirror under CC BY-SA 4.0.",
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert a raw ``RDD_SPLIT`` YOLO tree into the canonical COCO layout."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for source_split, target_split in _SPLIT_MAP.items():
            images_dir = raw_dir / source_split / "images"
            if not images_dir.is_dir():
                continue
            image_paths = sorted(
                path
                for path in images_dir.iterdir()
                if path.suffix.lower() in IMAGE_EXTENSIONS
            )
            _convert_split(image_paths, output_dir / target_split, source_split)


def _convert_split(
    image_paths: list[Path], split_output_dir: Path, split_name: str
) -> None:
    """Convert one YOLO split into a COCO split, dropping out-of-range class ids."""
    split_output_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    skipped_boxes = 0

    for image_id, image_path in enumerate(image_paths, start=1):
        label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        width, height = load_image_size(image_path)

        link_image(image_path, split_output_dir / image_path.name)
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )
        if not label_path.exists():
            continue

        for line in label_path.read_text(encoding="utf-8").splitlines():
            annotation = convert_yolo_label_line(
                line,
                LabelConversionContext(
                    image_id=image_id,
                    annotation_id=annotation_id,
                    image_width=width,
                    image_height=height,
                    category_count=len(_CLASSES),
                ),
            )
            if annotation is None:
                skipped_boxes += 1
                continue
            annotations.append(annotation)
            annotation_id += 1

    payload = {
        "info": {"description": f"RDD2022 canonical COCO ({split_output_dir.name})"},
        "licenses": [
            {
                "id": 1,
                "name": "CC BY-SA 4.0",
                "url": "https://creativecommons.org/licenses/by-sa/4.0/",
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
    print(
        f"[{split_name}] {len(images)} images, {len(annotations)} boxes, "
        f"skipped invalid boxes: {skipped_boxes}"
    )
