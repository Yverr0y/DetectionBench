"""
HRP4K (High-Resolution Pothole 4K) adapter -- road pothole detection.

Source: the official Zenodo release (``doi:10.5281/zenodo.17522874``) --
``{train,valid,test}/images/*.jpg`` plus already-COCO-format
``{train,valid,test}.json`` (one ``pothole`` category, id 0; boxes as COCO
``[x, y, w, h]``). ``valid``/``test`` are the official validation/test
splits and are used as-is.

**Known upstream data-completeness gap (train split only).** The official
``train.json`` references 4,203 unique images (ids 0-4202), but the
archive itself only actually ships 2,286 of them (ids 0-2285) --
confirmed against Zenodo's own file listing (exact byte-for-byte size
match with the published record), so this is not a corrupted download.
1,917 training images (and their 2,790 -> the surviving fraction of the
5,259 originally-referenced boxes) are simply absent from the V1.00
release. ``valid`` (900/900) and ``test`` (900/900) are complete. This
adapter filters every split down to images that actually exist on disk,
so a defective ``train.json`` never produces file-not-found errors or
orphan annotations -- it just yields a smaller-than-advertised train set.

License: CC BY 4.0, as stated on the Zenodo record.
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

_CLASSES = ["pothole"]

_SPLITS = {"train": "train", "valid": "valid", "test": "test"}


@register
class HRP4KAdapter(DatasetAdapter):
    """Adapter for the HRP4K road pothole-detection dataset."""

    spec = DatasetSpec(
        key="hrp4k",
        display_name="HRP4K",
        classes=_CLASSES,
        description=(
            "HRP4K (High-Resolution Pothole 4K) is a perspective-view road "
            "pothole-detection benchmark: 4,086 usable high-resolution road "
            "images (per this adapter's on-disk counts -- see module "
            "docstring for a known upstream train-split completeness gap) "
            "with 4,748 pothole instances, captured for automated "
            "road-condition assessment. It is a single-class, dense-detection "
            "style benchmark similar in spirit to RDD2022's pothole class, "
            "but pothole-specific and at much higher image resolution."
        ),
        homepage="https://github.com/hanshenChen/HRP4K",
        citation=(
            "@article{chen2026hrp4k,\n"
            "  title={A high-resolution perspective-view road image dataset "
            "for pothole detection},\n"
            "  author={Chen, Hanshen and Tu, Zhoulin and Zhao, Yu and Ye, "
            "Jianfeng},\n"
            "  journal={Scientific Data},\n"
            "  volume={13},\n"
            "  pages={961},\n"
            "  year={2026},\n"
            "  doi={10.1038/s41597-026-07317-w}\n"
            "}"
        ),
        license="CC BY 4.0 (Zenodo record doi:10.5281/zenodo.17522874).",
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert the official HRP4K release into canonical COCO splits."""
        for split in _SPLITS.values():
            if not (raw_dir / split / "images").is_dir():
                raise FileNotFoundError(f"Expected {raw_dir}/{split}/images.")
            if not (raw_dir / f"{split}.json").is_file():
                raise FileNotFoundError(f"Expected {raw_dir}/{split}.json.")
        output_dir.mkdir(parents=True, exist_ok=True)

        for split, out_name in _SPLITS.items():
            data = json.loads((raw_dir / f"{split}.json").read_text(encoding="utf-8"))
            _write_split(
                data["images"],
                data["annotations"],
                raw_dir / split / "images",
                output_dir / out_name,
            )


def _write_split(
    all_images: list[dict[str, Any]],
    all_annotations: list[dict[str, Any]],
    image_dir: Path,
    split_output_dir: Path,
) -> None:
    """Emit one canonical COCO split, dropping any image missing on disk."""
    split_output_dir.mkdir(parents=True, exist_ok=True)

    split_images = [im for im in all_images if (image_dir / im["file_name"]).exists()]
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
        link_image(image_dir / im["file_name"], split_output_dir / im["file_name"])
        images.append(
            {
                "id": im["id"],
                "file_name": im["file_name"],
                "width": im["width"],
                "height": im["height"],
            }
        )

    payload = {
        "info": {"description": f"HRP4K canonical COCO ({split_output_dir.name})"},
        "licenses": [
            {
                "id": 1,
                "name": "CC BY 4.0",
                "url": "https://creativecommons.org/licenses/by/4.0/",
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
