"""
SSDD (SAR Ship Detection Dataset) adapter.

Source: the official ``BBox_SSDD/coco_style`` release (Official-SSDD-OPEN,
Zhang et al. 2021) -- ``images/{train,test}/*.jpg`` (928 train + 232 test,
1,160 total) plus already-COCO-format ``annotations/{train,test}.json``
(one ``ship`` category, id 0; boxes as COCO ``[x, y, w, h]``; polygon
segmentation is present but unused for detection). ``test_inshore``/
``test_offshore`` are filtered *subsets* of ``test``, not separate splits,
and are not converted separately.

SSDD's own split (train 928 / test 232) is used as-is; this adapter carves
a seeded validation set out of train (``_VAL_FRACTION``), the same approach
used for HRSID (a SAR ship-detection dataset in the same situation: no
official validation split).

License: the GitHub repo (TianwenZhang0825/Official-SSDD) carries an
explicit **Apache-2.0** ``LICENSE`` (confirmed via GitHub's own license
detection, not just a badge claim) -- a real, unambiguous grant, unlike
HRSID's software-only GPL-3.0. However, SSDD's imagery is composited from
**RadarSat-2, TerraSAR-X, and Sentinel-1** -- the same second-order
sensor-rights situation as HRSID (TerraSAR-X/TanDEM-X is DLR
scientific-use, not automatically freely redistributable, independent of
what the repackager's own repo license grants). Fine to build/evaluate
against locally. Mirrored on Hugging Face tagged with the repo's own
Apache-2.0 license, with the TerraSAR-X caveat documented on the card --
see the dataset card for the full explanation.

Stats: see docs/datasets/ssdd/README.md (class distribution, split
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

_VAL_FRACTION = 0.15
_SPLIT_SEED = 42


@register
class SSDDAdapter(DatasetAdapter):
    """Adapter for the SSDD SAR ship-detection dataset."""

    spec = DatasetSpec(
        key="ssdd",
        display_name="SSDD",
        classes=_CLASSES,
        description=(
            "SSDD (SAR Ship Detection Dataset) is a benchmark for ship detection in "
            "Synthetic Aperture Radar imagery: 1,160 images with 2,587 ship "
            "instances, composited from RadarSat-2, TerraSAR-X, and Sentinel-1 at "
            "resolutions from 1m to 15m, across multiple polarizations and both "
            "inshore and offshore scenes. It's used to benchmark SAR ship "
            "detection, where speckle noise "
            "and side-lobe artifacts make optical-trained detectors unreliable -- the "
            "same problem HRSID targets, from a different sensor mix."
        ),
        homepage="https://github.com/TianwenZhang0825/Official-SSDD",
        citation=(
            "@article{zhang2021sar,\n"
            "  title={SAR Ship Detection Dataset (SSDD): Official Release and "
            "Comprehensive Data Analysis},\n"
            "  author={Zhang, Tianwen and Zhang, Xiaoling and Li, Jianwei and Xu, "
            "Xiaowo and Wang, Baoyou and Zhan, Xu and Xu, Yanqin and Ke, Xu and "
            "Zeng, Tianjiao and Su, Hao and others},\n"
            "  journal={Remote Sensing},\n"
            "  volume={13},\n"
            "  number={18},\n"
            "  pages={3690},\n"
            "  year={2021}\n"
            "}"
        ),
        license=(
            "Apache-2.0 (repo); imagery partly TerraSAR-X/TanDEM-X (DLR-restricted)."
        ),
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert the official SSDD coco_style release into canonical COCO."""
        image_dir = raw_dir / "images"
        annotation_dir = raw_dir / "annotations"
        if not image_dir.is_dir() or not annotation_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {raw_dir}/images and {raw_dir}/annotations."
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        train_data = json.loads(
            (annotation_dir / "train.json").read_text(encoding="utf-8")
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
            (annotation_dir / "test.json").read_text(encoding="utf-8")
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
    """Emit one canonical COCO split from a subset of an SSDD COCO file."""
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
                "category_id": 0,
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
        "info": {"description": f"SSDD canonical COCO ({split_output_dir.name})"},
        "licenses": [
            {
                "id": 1,
                "name": "Apache-2.0",
                "url": "https://www.apache.org/licenses/LICENSE-2.0",
            }
        ],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": _CLASSES[0], "supercategory": "none"}],
    }
    (split_output_dir / COCO_ANNOTATION_FILENAME).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"[{split_output_dir.name}] {len(images)} images, {len(annotations)} boxes")
