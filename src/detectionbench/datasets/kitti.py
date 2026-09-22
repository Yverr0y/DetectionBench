"""
KITTI adapter -- 2D object detection (Car/Pedestrian/Cyclist and 5 more).

Source: the official KITTI 2D object detection benchmark (Geiger, Lenz,
Urtasun, CVPR 2012) -- ``training/image_2/*.png`` (the official
"left color images" release, registration-gated at
https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)
plus ``training/label_2/*.txt``, one native KITTI-format label file per
image (whitespace-separated: ``type truncated occluded alpha left top
right bottom h w l x y z ry``; bbox is absolute-pixel ``[left, top, right,
bottom]``). Only the 7,481 officially labelled images are usable at all --
KITTI's ~7,518 test images have never had public ground truth.

**No official train/validation split exists.** This adapter uses the split
introduced by Chen, Kundu, Zhu, Berneshawi, Ma, Fidler, and Urtasun ("3DOP",
NeurIPS 2015) -- ``train`` 3,712 images / ``valid`` 3,769 images, the de
facto standard across the KITTI 3D/2D detection literature (used by
OpenPCDet, MMDetection3D, avod, SECOND, PointRCNN, and still the split used
in papers as recent as 2024-2025). The exact id lists are vendored in
``data/kitti_train_ids.txt`` / ``data/kitti_val_ids.txt`` (one 6-digit
image id per line, verified zero overlap, union equals all 7,481 labelled
images) -- sourced from the copy in the OpenPCDet repository, which itself
distributes Chen et al.'s original split. There is deliberately no
held-out ``test`` split here: the official test images have no public
labels to evaluate against, so all labelled data is used for train/val,
and ``valid`` is what "KITTI val" means in the wider literature.

Categories: KITTI's label files also mark ``DontCare`` regions -- an
ignore-region marker, not a real object class (ambiguous/unlabelled areas
evaluators should not penalize) -- dropped here, not treated as a class.
The remaining 8 classes are kept in their natural release order: Car,
Cyclist, Misc, Pedestrian, Person_sitting, Tram, Truck, Van.

License: CC BY-NC-SA 3.0, as stated on the official KITTI page
(https://www.cvlibs.net/datasets/kitti/).

Stats: see docs/datasets/kitti/README.md (class distribution, split
summary, box geometry -- generated via detectionbench-dataset-stats).
"""

from __future__ import annotations

import json
from importlib import resources
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

_CLASSES = [
    "Car",
    "Cyclist",
    "Misc",
    "Pedestrian",
    "Person_sitting",
    "Tram",
    "Truck",
    "Van",
]
_CLASS_TO_ID = {name: i for i, name in enumerate(_CLASSES)}
_IGNORED_TYPE = "DontCare"

_SPLIT_FILES = {"train": "kitti_train_ids.txt", "valid": "kitti_val_ids.txt"}


@register
class KITTIAdapter(DatasetAdapter):
    """Adapter for the KITTI 2D object detection dataset."""

    spec = DatasetSpec(
        key="kitti",
        display_name="KITTI",
        classes=_CLASSES,
        description=(
            "KITTI is a foundational autonomous-driving benchmark: street "
            "scenes captured from a moving vehicle in and around Karlsruhe, "
            "Germany, annotated for 2D object detection across 8 classes "
            "(Car, Cyclist, Misc, Pedestrian, Person_sitting, Tram, Truck, "
            "Van). Only the officially labelled 7,481 images are usable "
            "(test-set boxes are not public); this adapter splits them "
            "train/valid using the Chen et al. (2015) 3712/3769 split, the "
            "de facto standard used across the KITTI detection literature."
        ),
        homepage="https://www.cvlibs.net/datasets/kitti/",
        citation=(
            "@inproceedings{geiger2012kitti,\n"
            "  title={Are we ready for autonomous driving? The KITTI vision "
            "benchmark suite},\n"
            "  author={Geiger, Andreas and Lenz, Philip and Urtasun, Raquel},\n"
            "  booktitle={2012 IEEE Conference on Computer Vision and Pattern "
            "Recognition},\n"
            "  pages={3354--3361},\n"
            "  year={2012},\n"
            "  organization={IEEE},\n"
            "  doi={10.1109/CVPR.2012.6248074}\n"
            "}"
        ),
        license="CC BY-NC-SA 3.0 (official KITTI terms).",
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert the official KITTI release into canonical COCO train/valid splits."""
        image_dir = raw_dir / "training" / "image_2"
        label_dir = raw_dir / "training" / "label_2"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {raw_dir}/training/image_2 and {raw_dir}/training/label_2."
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        for split, filename in _SPLIT_FILES.items():
            ids = _load_split_ids(filename)
            _write_split(ids, image_dir, label_dir, output_dir / split)


def _load_split_ids(filename: str) -> list[str]:
    text = (resources.files("detectionbench.datasets") / "data" / filename).read_text()
    return sorted(line.strip() for line in text.splitlines() if line.strip())


def _parse_label_file(path: Path) -> list[tuple[str, list[float]]]:
    boxes: list[tuple[str, list[float]]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split()
        object_type = fields[0]
        if object_type == _IGNORED_TYPE or object_type not in _CLASS_TO_ID:
            continue
        left, top, right, bottom = (float(v) for v in fields[4:8])
        boxes.append((object_type, [left, top, right - left, bottom - top]))
    return boxes


def _write_split(
    ids: list[str], image_dir: Path, label_dir: Path, split_output_dir: Path
) -> None:
    """Emit one canonical COCO split from a list of 6-digit KITTI image ids."""
    split_output_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for image_id, stem in enumerate(ids):
        file_name = f"{stem}.png"
        src = image_dir / file_name
        if not src.exists():
            continue
        with Image.open(src) as im:
            width, height = im.size
        link_image(src, split_output_dir / file_name)
        images.append(
            {"id": image_id, "file_name": file_name, "width": width, "height": height}
        )
        for object_type, (x, y, box_width, box_height) in _parse_label_file(
            label_dir / f"{stem}.txt"
        ):
            if box_width <= 0 or box_height <= 0:
                continue
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": _CLASS_TO_ID[object_type],
                    "bbox": [x, y, box_width, box_height],
                    "area": box_width * box_height,
                    "segmentation": [],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    payload = {
        "info": {"description": f"KITTI canonical COCO ({split_output_dir.name})"},
        "licenses": [
            {
                "id": 1,
                "name": "CC BY-NC-SA 3.0",
                "url": "https://creativecommons.org/licenses/by-nc-sa/3.0/",
            }
        ],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i, "name": name, "supercategory": "none"}
            for i, name in enumerate(_CLASSES)
        ],
    }
    (split_output_dir / COCO_ANNOTATION_FILENAME).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"[{split_output_dir.name}] {len(images)} images, {len(annotations)} boxes")
