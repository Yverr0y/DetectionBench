"""
Global Wheat Head Dataset (GWHD) 2021 adapter.

GWHD 2021 (the competition release) ships as a flat ``images/`` directory
of 1024x1024 PNGs plus one CSV per split
(``competition_{train,val,test}.csv``) with columns::

    image_name, BoxesString, domain

``BoxesString`` holds all boxes for that image as a single
semicolon-separated string, each box itself a space-separated
``"x1 y1 x2 y2"`` in absolute pixel coordinates (top-left/bottom-right,
not normalized, not width/height). Images with no wheat heads use the
literal sentinel ``"no_box"`` instead of a box list. ``domain`` identifies
the contributing institution/field site (metadata only, not used here).
See https://www.global-wheat.com/ and ``metadata_dataset.csv`` in the raw
download for per-domain country/growth-stage info.

Stats: see docs/datasets/gwhd/README.md (class distribution, split
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

_CLASSES = ["wheat_head"]
_NO_BOX_SENTINEL = "no_box"
_BOX_FIELD_COUNT = 4

# GWHD's own split names differ from the canonical roboflow-style
# {train,valid,test} convention used elsewhere in this codebase.
_SPLIT_MAP = {"train": "train", "val": "valid", "test": "test"}
_CSV_FILENAMES = {
    "train": "competition_train.csv",
    "val": "competition_val.csv",
    "test": "competition_test.csv",
}


@register
class GWHDAdapter(DatasetAdapter):
    """Adapter for the Global Wheat Head Dataset (2021 competition release)."""

    spec = DatasetSpec(
        key="gwhd",
        display_name="Global Wheat Head Dataset",
        classes=_CLASSES,
        description=(
            "GWHD 2021 is a dense, single-class benchmark for wheat head localization, "
            "assembled from field images captured across multiple countries and "
            "research institutions to maximize genotype, growth-stage, and imaging- "
            "condition diversity. It's used to benchmark wheat head detection methods "
            "that generalize across environments -- a key requirement for real-world "
            "agricultural phenotyping and yield estimation."
        ),
        homepage="https://www.global-wheat.com/",
        citation=(
            "@article{david2021global,\n"
            "  title = {Global Wheat Head Dataset 2021: more diversity to improve the benchmarking of wheat head localization methods},\n"  # noqa: E501
            "  author = {David, Etienne and Serouart, Mario and Smith, Daniel and Madec, Simon and Velumani, Kaaviya and Liu, Shouyang and Wang, Xu and Pinto Espinosa, Francisco and Shafiee, Shahameh and Tahir, Izzat S. A. and Tsujimoto, Hisashi and Nasuda, Shuhei and Zheng, Bangyou and Kichgessner, Norbert and Aasen, Helge and Hund, Andreas and Sadhegi-Tehran, Pouria and Nagasawa, Koichi and Ishikawa, Goro and Dandrifosse, S{\\'e}bastien and Carlier, Alexis and Mercatoris, Benoit and Kuroki, Ken and Wang, Haozhou and Ishii, Masanori and Badhon, Minhajul A. and Pozniak, Curtis and LeBauer, David Shaner and Lilimo, Morten and Poland, Jesse and Chapman, Scott and de Solan, Benoit and Baret, Fr{\\'e}d{\\'e}ric and Stavness, Ian and Guo, Wei},\n"  # noqa: E501
            "  journal = {Plant Phenomics},\n"
            "  year = {2021},\n"
            "  doi = {10.34133/2021/9846158}\n"
            "}"
        ),
        license="CC BY 4.0 -- see homepage before redistributing.",
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert a raw GWHD 2021 download into the canonical COCO layout."""
        images_dir = raw_dir / "images"
        output_dir.mkdir(parents=True, exist_ok=True)

        for source_split, target_split in _SPLIT_MAP.items():
            csv_path = raw_dir / _CSV_FILENAMES[source_split]
            if not csv_path.exists():
                continue
            _convert_split(csv_path, images_dir, output_dir / target_split)


def _convert_split(csv_path: Path, images_dir: Path, split_output_dir: Path) -> None:
    """Convert one GWHD CSV split into the canonical layout."""
    split_output_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    seen_names: set[str] = set()

    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for image_id, row in enumerate(reader, start=1):
            source_name = row["image_name"]
            # A handful of GWHD 2021 CSV rows reuse the same image_name with
            # different domain/box data (an upstream data quirk, not a
            # duplicate to drop) -- disambiguate so each row keeps its own
            # image entry instead of two ids aliasing one file_name.
            file_name = _unique_output_name(source_name, seen_names)
            src = images_dir / source_name
            dst = split_output_dir / file_name
            if src.exists():
                link_image(src, dst)

            width, height = _image_size(src)
            images.append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                }
            )

            for bbox in _parse_boxes(row["BoxesString"]):
                x1, y1, x2, y2 = bbox
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                if bbox_width <= 0 or bbox_height <= 0:
                    continue
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 0,
                        "bbox": [x1, y1, bbox_width, bbox_height],
                        "area": bbox_width * bbox_height,
                        "segmentation": [],
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

    payload = {
        "info": {"description": f"Converted from GWHD split '{csv_path.stem}'"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": _CLASSES[0], "supercategory": "none"}],
    }
    (split_output_dir / COCO_ANNOTATION_FILENAME).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _parse_boxes(boxes_string: str) -> list[tuple[float, float, float, float]]:
    """Parse a GWHD ``BoxesString`` cell into a list of (x1, y1, x2, y2) boxes."""
    boxes_string = boxes_string.strip()
    if not boxes_string or boxes_string.lower() == _NO_BOX_SENTINEL:
        return []

    boxes: list[tuple[float, float, float, float]] = []
    for box_str in boxes_string.split(";"):
        parts = box_str.split()
        if len(parts) != _BOX_FIELD_COUNT:
            continue
        x1, y1, x2, y2 = map(float, parts)
        boxes.append((x1, y1, x2, y2))
    return boxes


def _image_size(image_path: Path) -> tuple[int, int]:
    """Read image width and height without keeping the file open."""
    with Image.open(image_path) as image:
        return image.size


def _unique_output_name(name: str, seen_names: set[str]) -> str:
    """Disambiguate a repeated image_name within one split's CSV."""
    if name not in seen_names:
        seen_names.add(name)
        return name

    stem, suffix = Path(name).stem, Path(name).suffix
    counter = 1
    while True:
        candidate = f"{stem}__{counter}{suffix}"
        if candidate not in seen_names:
            seen_names.add(candidate)
            return candidate
        counter += 1
