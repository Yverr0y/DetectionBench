"""
PubLayNet (document layout analysis) adapter.

Source: the official IBM release -- ``{train,val}.json`` (COCO-format
annotations, 5 layout classes) plus ``{train,val}/*.jpg`` page-image
directories. ``test.json`` exists upstream but was a blind, unlabeled
competition set (ICDAR 2021) with no public ground truth, matching this
project's precedent for SeaDronesSee -- so only ``train``/``val`` are
converted; this adapter carves a seeded validation slice out of neither
(the official ``val`` split, 11,245 images, is used as-is for evaluation).

IBM's original Data Asset eXchange (DAX) hosting for PubLayNet is
deprecated; the README now points at third-party Hugging Face / Kaggle
re-uploads. Those Hugging Face copies are large (~100-200 GB, parquet with
embedded image bytes, all 335,703 train images at full page resolution) --
fetch and stage them into this adapter's expected
``{train,val}.json`` + ``{train,val}/*.jpg`` layout yourself before running
``prepare_coco``; this adapter does not fetch or convert parquet directly.

Classes: text, title, list, table, figure (the fixed 5-class PubLayNet
taxonomy; source category ids are 1-5, remapped to 0-4).

License: the annotations are IBM's, licensed CDLA-Permissive-1.0 (same as
DocLayNet -- commercial use permitted, attribution required). IBM
explicitly does not own the copyright of the page images themselves; those
are governed by the PubMed Central Open Access Subset's own terms (which
only includes articles carrying an open license as a condition of
inclusion in that subset).
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

_CLASSES = ["text", "title", "list", "table", "figure"]

# PubLayNet's own split file (stem) -> canonical split name. ``test`` is
# deliberately excluded -- no public ground truth (see module docstring).
_SPLIT_FILES = {"train": "train", "val": "valid"}


@register
class PubLayNetAdapter(DatasetAdapter):
    """Adapter for IBM's PubLayNet document-layout-analysis dataset."""

    spec = DatasetSpec(
        key="publaynet",
        display_name="PubLayNet",
        classes=_CLASSES,
        homepage="https://github.com/ibm-aur-nlp/PubLayNet",
        citation=(
            "Zhong, Tang, Yepes, 'PubLayNet: largest dataset ever for "
            "document layout analysis', ICDAR 2019."
        ),
        license=(
            "CDLA-Permissive-1.0 (annotations, IBM); page images governed "
            "by the PMC Open Access Subset's own terms."
        ),
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert a staged PubLayNet release into the canonical COCO layout."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for source_split, target_split in _SPLIT_FILES.items():
            json_path = raw_dir / f"{source_split}.json"
            image_dir = raw_dir / source_split
            if not json_path.exists():
                continue
            _convert_split(json_path, image_dir, output_dir / target_split)


def _convert_split(json_path: Path, image_dir: Path, split_output_dir: Path) -> None:
    """Reshuffle one PubLayNet COCO split into the canonical layout."""
    with json_path.open(encoding="utf-8") as file:
        data = json.load(file)

    categories = sorted(data["categories"], key=lambda category: category["id"])
    cat_id_to_idx = {category["id"]: index for index, category in enumerate(categories)}

    split_output_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict[str, Any]] = []
    for image in data["images"]:
        src = image_dir / image["file_name"]
        dst = split_output_dir / image["file_name"]
        if src.exists():
            link_image(src, dst)
        images.append(image)

    annotations = [
        {**annotation, "category_id": cat_id_to_idx[annotation["category_id"]]}
        for annotation in data["annotations"]
        if annotation["category_id"] in cat_id_to_idx
    ]
    remapped_categories = [
        {
            "id": index,
            "name": category["name"],
            "supercategory": category.get("supercategory", "none"),
        }
        for index, category in enumerate(categories)
    ]

    payload = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": remapped_categories,
    }
    (split_output_dir / COCO_ANNOTATION_FILENAME).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"[{split_output_dir.name}] {len(images)} images, {len(annotations)} boxes")
