"""
UAVDT (UAV Detection and Tracking benchmark) adapter -- detection subset.

Source: the Dataset Ninja export of UAVDT, which ships as Supervisely-format
per-image JSON -- ``{train,test}/img/*.jpg`` plus
``{train,test}/ann/<image>.jpg.json`` (axis-aligned ``rectangle`` objects
under ``objects[].points.exterior`` as ``[[x1, y1], [x2, y2]]``; image size
under ``size``; per-image sequence / weather / altitude tags under ``tags``).

UAVDT's own split is train (30 sequences) / test (70 sequences), with no
validation set. This adapter keeps ``test`` as-is and carves a **sequence-
aware** validation split out of ``train`` (``_VAL_SEQUENCE_FRACTION`` of the
30 training sequences, chosen by a seeded shuffle) so consecutive video
frames from one sequence never straddle the train/val boundary.

Classes: the canonical UAVDT-DET taxonomy is ``car`` / ``truck`` / ``bus``
(the Dataset Ninja ``vehicle`` class is unused in practice); any object
outside these three is dropped.

License: UAVDT is distributed "for research purpose only" with no
redistribution grant -- fine to build/evaluate against locally, **not** to
re-host. No Hugging Face mirror without the authors' written permission.

Stats: see docs/datasets/uavdt/README.md (class distribution, per-sequence
breakdown, box geometry -- generated via detectionbench-dataset-stats).
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

_CLASSES = ["car", "truck", "bus"]
_CLASS_INDEX = {name: index for index, name in enumerate(_CLASSES)}

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_RECTANGLE = "rectangle"
_BOX_CORNER_COUNT = 2

# Raw split dir -> canonical split(s). ``train`` is further divided below.
_RAW_SPLITS = ("train", "test")
_VAL_SEQUENCE_FRACTION = 0.2
_SPLIT_SEED = 42


def _sequence_of(image_name: str) -> str:
    """UAVDT frame name ``<SEQ>_img<NNNNNN>.jpg`` -> owning sequence id ``<SEQ>``."""
    return image_name.split("_img", 1)[0]


@register
class UAVDTAdapter(DatasetAdapter):
    """Adapter for the UAVDT aerial vehicle detection benchmark (Supervisely export)."""

    spec = DatasetSpec(
        key="uavdt",
        display_name="UAVDT",
        classes=_CLASSES,
        description=(
            "UAVDT (UAV Detection and Tracking) is a large-scale benchmark for "
            "vehicle detection, single-object tracking, and multi-object "
            "tracking from footage captured by an unmanned aerial vehicle. "
            "It consists of about 80,000 representative frames from 100 video "
            "sequences, densely annotated with car/truck/bus bounding boxes "
            "plus per-sequence attributes (weather, flying altitude, camera "
            "view, vehicle category, occlusion). It's used to study detection "
            "and tracking under conditions aerial traffic-surveillance systems "
            "actually face: small objects, dense traffic, occlusion, and "
            "large viewpoint/altitude changes -- this adapter uses the "
            "detection-subset labels only."
        ),
        homepage="https://sites.google.com/view/grli-uavdt",
        citation=(
            "@InProceedings{du2018unmanned,\n"
            "  title={The Unmanned Aerial Vehicle Benchmark: Object Detection "
            "and Tracking},\n"
            "  author={Du, Dawei and Qi, Yuankai and Yu, Hongyang and Yang, "
            "Yifan and Duan, Kaiwen and Li, Guorong and Zhang, Weigang and "
            "Huang, Qingming and Tian, Qi},\n"
            "  booktitle={Proceedings of the European Conference on Computer "
            "Vision (ECCV)},\n"
            "  year={2018}\n"
            "}"
        ),
        license="Research use only -- no redistribution; see homepage.",
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert the UAVDT Dataset Ninja export into the canonical COCO layout."""
        for split in _RAW_SPLITS:
            if not (raw_dir / split / "img").is_dir():
                raise FileNotFoundError(
                    f"Expected a Dataset Ninja layout at {raw_dir}/{split}/(img|ann)."
                )
        output_dir.mkdir(parents=True, exist_ok=True)

        train_images = sorted(
            p
            for p in (raw_dir / "train" / "img").iterdir()
            if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        sequences = sorted({_sequence_of(p.name) for p in train_images})
        random.Random(_SPLIT_SEED).shuffle(sequences)  # noqa: S311  # nosec: B311
        n_val = max(1, round(len(sequences) * _VAL_SEQUENCE_FRACTION))
        val_sequences = set(sequences[:n_val])

        members = {
            "train": [
                p for p in train_images if _sequence_of(p.name) not in val_sequences
            ],
            "valid": [p for p in train_images if _sequence_of(p.name) in val_sequences],
            "test": sorted(
                p
                for p in (raw_dir / "test" / "img").iterdir()
                if p.suffix.lower() in _IMAGE_EXTENSIONS
            ),
        }
        raw_split_of = {"train": "train", "valid": "train", "test": "test"}
        for target_split, image_paths in members.items():
            _convert_split(
                image_paths,
                raw_dir / raw_split_of[target_split] / "ann",
                output_dir / target_split,
            )


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
            class_id = _CLASS_INDEX.get(obj["classTitle"])
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
        "info": {"description": f"UAVDT canonical COCO ({split_output_dir.name})"},
        "licenses": [{"id": 1, "name": "Research use only", "url": ""}],
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
