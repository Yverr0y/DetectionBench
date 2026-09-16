"""
LLVIP (visible-infrared paired pedestrian detection) adapter.

Source: the official LLVIP release, a registered visible/infrared image-pair
dataset with a single shared Pascal VOC XML annotation set:

    raw_dir/
    ├── infrared/{train,test}/*.jpg
    ├── visible/{train,test}/*.jpg    (not used by this adapter)
    └── Annotations/*.xml             (one VOC XML per frame, shared by
                                        both modalities since the pairs are
                                        pixel-registered)

**This adapter uses the infrared images only.** LLVIP's own premise is that
infrared vastly outperforms visible-light imagery for pedestrian detection
in the low-light/nighttime conditions the dataset was captured in -- the
paired visible frames are frequently near-black and not independently
useful for detection. Pairing the two modalities into a joint RGB+IR
detector is a real, separate research direction DetectionBench does not
attempt here (its training/eval pipeline is single-image); this adapter
picks the modality that is actually usable on its own.

If your download uses different raw folder names (some LLVIP mirrors use
``Train_LLVIP_ir``/``Test_LLVIP_ir`` instead of ``infrared/{train,test}``),
adjust ``_SPLIT_DIRS`` below -- this has not been verified against every
LLVIP redistribution, only the layout documented in the official README.

LLVIP ships train/test only, no official validation split; this adapter
carves a seeded 15% slice out of train (same approach as the HRSID and DUO
adapters).

License: LLVIP has a real, explicit license (``Term of Use and License.md``
in the official repo) -- free for **non-commercial** academic and personal
use with attribution; it reserves "all rights not expressly granted" and
does not grant third-party redistribution of the raw dataset. Fine to
build/evaluate against locally, **not** to re-host. No Hugging Face mirror
without the authors' written permission -- see
``detectionbench-download-dataset --dataset llvip`` for the official
download locations instead.

Stats: see docs/datasets/llvip/README.md (class distribution, split
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

_CLASSES = ["person"]
_CLASS_INDEX = {"person": 0}

# Raw split dir (under infrared/) -> handling. ``train`` is further split
# into train/valid below (LLVIP has no official validation set).
_SPLIT_DIRS = {"train": "infrared/train", "test": "infrared/test"}
_ANNOTATION_DIR = "Annotations"

_VAL_FRACTION = 0.15
_SPLIT_SEED = 42


@register
class LLVIPAdapter(DatasetAdapter):
    """Adapter for the LLVIP visible-infrared paired pedestrian-detection dataset."""

    spec = DatasetSpec(
        key="llvip",
        display_name="LLVIP",
        classes=_CLASSES,
        description=(
            "LLVIP is a registered visible/infrared image-pair dataset for pedestrian "
            "detection in low-light conditions: 30,976 image pairs (paired visible + "
            "infrared frames, pixel-aligned) captured at night, annotated with "
            "pedestrian bounding boxes. Its central finding is that infrared imagery "
            "is dramatically more useful than visible imagery under these conditions "
            "-- this adapter uses the infrared images only."
        ),
        homepage="https://github.com/bupt-ai-cz/LLVIP",
        citation=(
            "@inproceedings{jia2021llvip,\n"
            "  title={LLVIP: A Visible-infrared Paired Dataset for Low-light Vision},\n"
            "  author={Jia, Xinyu and Zhu, Chuang and Li, Minzhen and Tang, Wenqi and Zhou, Wenli},\n"  # noqa: E501
            "  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops},\n"  # noqa: E501
            "  pages={3496--3504},\n"
            "  year={2021}\n"
            "}"
        ),
        license=(
            "Non-commercial academic/personal use only, attribution "
            "required, no redistribution grant. See homepage."
        ),
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert the LLVIP infrared split into the canonical COCO layout."""
        annotation_dir = raw_dir / _ANNOTATION_DIR
        train_dir = raw_dir / _SPLIT_DIRS["train"]
        if not annotation_dir.is_dir() or not train_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {raw_dir}/{_ANNOTATION_DIR} and "
                f"{raw_dir}/{_SPLIT_DIRS['train']}. If your download uses "
                "different folder names, edit _SPLIT_DIRS in this module."
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        train_images = sorted(train_dir.glob("*.jpg"))
        shuffled = train_images[:]
        random.Random(_SPLIT_SEED).shuffle(shuffled)  # noqa: S311  # nosec: B311
        n_val = max(1, round(len(shuffled) * _VAL_FRACTION))
        val_stems = {p.stem for p in shuffled[:n_val]}

        _convert_split(
            [p for p in train_images if p.stem not in val_stems],
            annotation_dir,
            output_dir / "train",
        )
        _convert_split(
            [p for p in train_images if p.stem in val_stems],
            annotation_dir,
            output_dir / "valid",
        )
        test_dir = raw_dir / _SPLIT_DIRS["test"]
        if test_dir.is_dir():
            _convert_split(
                sorted(test_dir.glob("*.jpg")), annotation_dir, output_dir / "test"
            )


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
        "info": {"description": f"LLVIP canonical COCO ({split_output_dir.name})"},
        "licenses": [{"id": 1, "name": "See dataset homepage", "url": ""}],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": "person", "supercategory": "none"}],
    }
    (split_output_dir / COCO_ANNOTATION_FILENAME).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"[{split_output_dir.name}] {len(images)} images, {len(annotations)} boxes")
