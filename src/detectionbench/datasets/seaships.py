"""
SeaShips (7000) adapter -- maritime ship detection.

Source: the official SeaShips(7000) Pascal-VOC-style release --
``JPEGImages/*.jpg`` (7,000 1920x1080 frames from a coastline surveillance
system) plus ``Annotations/*.xml`` (one Pascal VOC XML per image) and
``ImageSets/Main/{train,val,test,trainval}.txt`` (the official split --
1,750 / 1,750 / 3,500 images, used as-is; ``trainval.txt`` is redundant
with ``train``+``val`` combined and not read separately).

Classes: the paper's fixed 6-class taxonomy -- ore carrier, bulk cargo
carrier, general cargo ship, container ship, fishing boat, passenger ship.

License: unclear, no explicit redistribution grant. The original host
(``lmars.whu.edu.cn/prof_web/...``) is defunct (the lab's site was
restructured to ``liesmars.whu.edu.cn`` and the old path now redirects to
its homepage); the official GitHub repo (jiaming-wang/SeaShips) offers only
that dead link plus a gated Baidu Netdisk link, and states no explicit
license. Fine to build/evaluate against locally, **not** to re-host. No
Hugging Face mirror without the authors' written permission.
"""

from __future__ import annotations

import json
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
    "ore carrier",
    "bulk cargo carrier",
    "general cargo ship",
    "container ship",
    "fishing boat",
    "passenger ship",
]
_CLASS_INDEX = {name: index for index, name in enumerate(_CLASSES)}

# ImageSets/Main split file (stem) -> canonical split name.
_SPLIT_FILES = {"train": "train", "val": "valid", "test": "test"}


@register
class SeaShipsAdapter(DatasetAdapter):
    """Adapter for the SeaShips(7000) maritime ship-detection dataset."""

    spec = DatasetSpec(
        key="seaships",
        display_name="SeaShips",
        classes=_CLASSES,
        homepage="https://github.com/jiaming-wang/SeaShips",
        citation=(
            "Shao et al., 'SeaShips: A Large-Scale Precisely Annotated "
            "Dataset for Ship Detection', IEEE Transactions on Multimedia, "
            "20(10):2593-2604, 2018."
        ),
        license="Unclear -- no explicit grant; original host defunct. See homepage.",
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert the SeaShips(7000) VOC release into the canonical COCO layout."""
        image_sets_dir = raw_dir / "ImageSets" / "Main"
        annotation_dir = raw_dir / "Annotations"
        image_dir = raw_dir / "JPEGImages"
        if not image_sets_dir.is_dir() or not annotation_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {raw_dir}/ImageSets/Main, {raw_dir}/Annotations, "
                f"and {raw_dir}/JPEGImages."
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        for split_file, target_split in _SPLIT_FILES.items():
            list_path = image_sets_dir / f"{split_file}.txt"
            if not list_path.exists():
                continue
            stems = [
                line.strip()
                for line in list_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            _convert_split(stems, image_dir, annotation_dir, output_dir / target_split)


def _convert_split(
    stems: list[str], image_dir: Path, annotation_dir: Path, split_output_dir: Path
) -> None:
    """Parse one split's Pascal VOC XML annotations into a canonical COCO JSON."""
    split_output_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1

    for image_id, stem in enumerate(stems, start=1):
        xml_path = annotation_dir / f"{stem}.xml"
        image_path = image_dir / f"{stem}.jpg"
        if not xml_path.exists() or not image_path.exists():
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
        "info": {"description": f"SeaShips canonical COCO ({split_output_dir.name})"},
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
