"""
CeyMo adapter -- road-marking detection.

Source: the official CeyMo release (Jayasinghe et al., WACV 2022; GitHub
``oshadajay/CeyMo``, Google Drive ``train.zip`` / ``test.zip``) --
``{train,test}/images/*.jpg`` (1920x1080) plus per-image Pascal-VOC XML
boxes in ``{train,test}/bbox_annotations/*.xml`` (11 road-marking classes,
identified by two/three-letter codes). The release also ships polygon
JSON and PNG segmentation masks; only the bounding boxes are used here. The
test set's per-image scenario category (normal, crowded, dazzle light,
night, rain, shadow) lives only in the polygon JSON and is not carried over.

CeyMo has no official validation split, so this adapter keeps the official
``test`` set as ``test`` (788 images) and carves a seeded validation set out
of ``train`` (``_VAL_FRACTION``), the same approach used for HRSID/SSDD.

License: MIT (GitHub repo LICENSE, confirmed via GitHub's license API).

Stats: see docs/datasets/ceymo/README.md (class distribution, split
summary, box geometry -- generated via detectionbench-dataset-stats).
"""

from __future__ import annotations

import json
import random
import xml.etree.ElementTree as ET  # noqa: S405  # nosec: B405 - trusted release
from pathlib import Path
from typing import Any

from detectionbench.datasets.base import (
    COCO_ANNOTATION_FILENAME,
    DatasetAdapter,
    DatasetSpec,
    link_image,
)
from detectionbench.datasets.registry import register

# Release code -> readable class name, in the release's own alphabetical-code order.
_CODE_TO_NAME = {
    "BL": "bus_lane",
    "CL": "cycle_lane",
    "DM": "diamond",
    "JB": "junction_box",
    "LA": "left_arrow",
    "PC": "pedestrian_crossing",
    "RA": "right_arrow",
    "SA": "straight_arrow",
    "SL": "slow",
    "SLA": "straight_left_arrow",
    "SRA": "straight_right_arrow",
}
_CLASSES = list(_CODE_TO_NAME.values())
_CODE_TO_ID = {code: i for i, code in enumerate(_CODE_TO_NAME)}

_VAL_FRACTION = 0.15
_SPLIT_SEED = 42


@register
class CeyMoAdapter(DatasetAdapter):
    """Adapter for the CeyMo road-marking detection dataset."""

    spec = DatasetSpec(
        key="ceymo",
        display_name="CeyMo",
        classes=_CLASSES,
        description=(
            "CeyMo is a road-marking detection benchmark from Sri Lanka: 2,887 "
            "1920x1080 road images with 4,706 road-marking instances across 11 "
            "classes (arrows, pedestrian crossings, bus/cycle lanes, junction "
            "boxes, diamonds, and 'slow' markings), covering urban, sub-urban "
            "and rural roads, with a test set spanning normal, crowded, dazzle "
            "light, night, rain and shadow conditions. Road markings are flat, "
            "perspective-distorted ground-plane objects, unlike the upright "
            "vehicles and signs most driving benchmarks target."
        ),
        homepage="https://github.com/oshadajay/CeyMo",
        github="https://github.com/oshadajay/CeyMo",
        citation=(
            "@InProceedings{Jayasinghe_2022_WACV,\n"
            "  title={CeyMo: See More on Roads - A Novel Benchmark Dataset for "
            "Road Marking Detection},\n"
            "  author={Jayasinghe, Oshada and Hemachandra, Sahan and Anhettigama, "
            "Damith and Kariyawasam, Shenali and Rodrigo, Ranga and Jayasekara, "
            "Peshala},\n"
            "  booktitle={Proceedings of the IEEE/CVF Winter Conference on "
            "Applications of Computer Vision (WACV)},\n"
            "  month={January},\n"
            "  year={2022},\n"
            "  pages={3104-3113}\n"
            "}"
        ),
        license="MIT (GitHub repo LICENSE).",
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert the official CeyMo release into canonical COCO splits."""
        for split in ("train", "test"):
            for sub in ("images", "bbox_annotations"):
                if not (raw_dir / split / sub).is_dir():
                    raise FileNotFoundError(f"Expected {raw_dir}/{split}/{sub}.")
        output_dir.mkdir(parents=True, exist_ok=True)

        train_stems = sorted(
            p.stem for p in (raw_dir / "train" / "bbox_annotations").glob("*.xml")
        )
        shuffled = train_stems[:]
        random.Random(_SPLIT_SEED).shuffle(shuffled)  # noqa: S311  # nosec: B311
        n_val = max(1, round(len(shuffled) * _VAL_FRACTION))
        val_stems = set(shuffled[:n_val])

        train_dir = raw_dir / "train"
        _write_split(
            [s for s in train_stems if s not in val_stems],
            train_dir,
            output_dir / "train",
        )
        _write_split(
            [s for s in train_stems if s in val_stems], train_dir, output_dir / "valid"
        )
        test_dir = raw_dir / "test"
        test_stems = sorted(
            p.stem for p in (test_dir / "bbox_annotations").glob("*.xml")
        )
        _write_split(test_stems, test_dir, output_dir / "test")


def _parse_xml(xml_path: Path) -> tuple[str, int, int, list[tuple[str, list[float]]]]:
    root = ET.parse(xml_path).getroot()  # noqa: S314  # nosec: B314 - trusted release
    file_name = root.findtext("filename") or f"{xml_path.stem}.jpg"
    width = int(root.findtext("size/width") or 0)
    height = int(root.findtext("size/height") or 0)
    boxes: list[tuple[str, list[float]]] = []
    for obj in root.findall("object"):
        code = (obj.findtext("name") or "").strip()
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        x1 = float(bnd.findtext("xmin") or 0)
        y1 = float(bnd.findtext("ymin") or 0)
        x2 = float(bnd.findtext("xmax") or 0)
        y2 = float(bnd.findtext("ymax") or 0)
        boxes.append((code, [x1, y1, x2 - x1, y2 - y1]))
    return file_name, width, height, boxes


def _write_split(stems: list[str], split_dir: Path, split_output_dir: Path) -> None:
    """Emit one canonical COCO split from a list of CeyMo image stems."""
    split_output_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for image_id, stem in enumerate(stems):
        file_name, width, height, boxes = _parse_xml(
            split_dir / "bbox_annotations" / f"{stem}.xml"
        )
        src = split_dir / "images" / file_name
        if not src.exists():
            continue
        link_image(src, split_output_dir / file_name)
        images.append(
            {"id": image_id, "file_name": file_name, "width": width, "height": height}
        )
        for code, (x, y, box_width, box_height) in boxes:
            if code not in _CODE_TO_ID or box_width <= 0 or box_height <= 0:
                continue
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": _CODE_TO_ID[code],
                    "bbox": [x, y, box_width, box_height],
                    "area": float(box_width) * float(box_height),
                    "segmentation": [],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    payload = {
        "info": {"description": f"CeyMo canonical COCO ({split_output_dir.name})"},
        "licenses": [
            {
                "id": 1,
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT",
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
