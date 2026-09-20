"""
PKLot adapter -- parking-space occupancy detection.

Source: the official PKLot release (Almeida et al., ESWA 2015; UFPR Vision,
Robotics and Imaging lab) -- ``PKLot/<lot>/<weather>/<date>/*.jpg`` (12,417
full 1280x720 frames from three fixed cameras: ``PUCPR``, ``UFPR04``,
``UFPR05``) with one XML per frame listing every marked parking space (a
rotated rectangle plus a 4-point contour) and an ``occupied`` flag. The
release's ``PKLotSegmented`` tree (~695k pre-cropped single-space patches)
is a classification variant and is not used.

**Framing caveat.** PKLot's own task is *per-space classification*: the
space locations are fixed per camera and only occupied/vacant is predicted.
This adapter instead treats every labelled space as a bounding box and asks a
detector to localise *and* classify it (``vacant`` / ``occupied``). Because
each camera never moves, spaces sit at near-identical pixel positions in
every frame, so a detector can partly memorise locations; scores are not
comparable to free-form car detection or to the classification literature.

Annotation handling: each space's box is the axis-aligned bounding box of its
contour polygon, clipped to the frame. 7,671 labelled spaces (1.1%, all in
UFPR04) have no contour in the release, only a rotated rectangle; for those
the box is the axis-aligned extent of the rotated rectangle. Where both exist
the two agree on average (mean IoU ~0.85) but the rotated-rect box is looser,
so those UFPR04 boxes are slightly less tight than the rest. Spaces with no
``occupied`` attribute (~25.8k of ~719k) are unlabelled and are dropped, not
guessed; they remain in the image as unlabelled background. One frame
(PUCPR/Sunny/2012-11-06 18_48_46) has no XML and is skipped.

Splits: PKLot has no official split, and frames are time-lapse captures from
fixed cameras, so a random image split would put near-duplicate frames on both
sides. This adapter groups frames by (lot, capture date -- taken from the
filename timestamp, since ~200 frames sit in a folder whose date differs) and
assigns whole groups to train/valid/test per lot (seeded, ~70/15/15 by image
count). No capture day spans two splits, but every lot appears in every split,
so this measures generalisation to unseen days on *seen* cameras, not to new
lots or new weather (rainy days are rare, so some splits may hold few).

License: CC BY 4.0 (stated on the official page and in the tarball's
``licence`` file).

Stats: see docs/datasets/pklot/README.md (class distribution, split
summary, box geometry -- generated via detectionbench-dataset-stats).
"""

from __future__ import annotations

import json
import math
import random
import xml.etree.ElementTree as ET  # noqa: S405  # nosec: B405 - trusted release
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

_CLASSES = ["vacant", "occupied"]  # index == the XML's ``occupied`` value

_TEST_FRACTION = 0.15
_VAL_FRACTION = 0.15
_SPLIT_SEED = 42
_MIN_CONTOUR_POINTS = 3

_Frame = tuple[str, Path, Path]  # (lot, jpg, xml)


@register
class PKLotAdapter(DatasetAdapter):
    """Adapter for the PKLot parking-space occupancy dataset."""

    spec = DatasetSpec(
        key="pklot",
        display_name="PKLot",
        classes=_CLASSES,
        description=(
            "PKLot is a parking-lot occupancy benchmark: 12,417 1280x720 frames "
            "from three fixed cameras (PUCPR, UFPR04, UFPR05) under sunny, "
            "cloudy and rainy conditions, with roughly 694k labelled parking "
            "spaces marked occupied or vacant. Its original task classifies "
            "each fixed space; here every space is a bounding box to detect "
            "and classify, so results are not comparable to free-form car "
            "detection."
        ),
        homepage="https://web.inf.ufpr.br/vri/databases/parking-lot-database/",
        citation=(
            "@article{almeida2015pklot,\n"
            "  title={PKLot -- A robust dataset for parking lot classification},\n"
            "  author={de Almeida, Paulo R. L. and Oliveira, Luiz S. and Britto "
            "Jr, Alceu S. and Silva Jr, Eunelson J. and Koerich, Alessandro L.},\n"
            "  journal={Expert Systems with Applications},\n"
            "  volume={42},\n"
            "  number={11},\n"
            "  pages={4937--4949},\n"
            "  year={2015},\n"
            "  doi={10.1016/j.eswa.2015.02.009}\n"
            "}"
        ),
        license="CC BY 4.0 (official page and tarball licence file).",
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert the official PKLot release into canonical COCO splits."""
        root = raw_dir / "PKLot"
        if not root.is_dir():
            raise FileNotFoundError(f"Expected {root} (the release's full-frame tree).")
        output_dir.mkdir(parents=True, exist_ok=True)

        groups, no_xml = _collect_groups(root)
        assignment = _assign_splits(groups)

        names: set[str] = set()
        dropped_unlabelled = dropped_degenerate = used_fallback = 0
        for split in ("train", "valid", "test"):
            frames = sorted(assignment[split], key=lambda f: (f[0], f[1].name))
            unlabelled, degenerate, fallback = _write_split(
                frames, output_dir / split, names
            )
            dropped_unlabelled += unlabelled
            dropped_degenerate += degenerate
            used_fallback += fallback
        print(
            f"skipped {no_xml} frame(s) with no XML; dropped {dropped_unlabelled} "
            f"unlabelled and {dropped_degenerate} degenerate space(s); "
            f"{used_fallback} box(es) derived from rotatedRect (no contour)"
        )


def _collect_groups(root: Path) -> tuple[dict[tuple[str, str], list[_Frame]], int]:
    """Group frames by (lot, filename date); count frames lacking an XML."""
    groups: dict[tuple[str, str], list[_Frame]] = {}
    no_xml = 0
    for lot_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for jpg in sorted(lot_dir.rglob("*.jpg")):
            xml = jpg.with_suffix(".xml")
            if not xml.exists():
                no_xml += 1
                continue
            groups.setdefault((lot_dir.name, jpg.name[:10]), []).append(
                (lot_dir.name, jpg, xml)
            )
    return groups, no_xml


def _assign_splits(
    groups: dict[tuple[str, str], list[_Frame]],
) -> dict[str, list[_Frame]]:
    """Assign whole (lot, date) groups to splits, per lot, by image count."""
    rng = random.Random(_SPLIT_SEED)  # noqa: S311  # nosec: B311
    assignment: dict[str, list[_Frame]] = {"train": [], "valid": [], "test": []}
    lots = sorted({lot for lot, _ in groups})
    for lot in lots:
        keys = sorted(k for k in groups if k[0] == lot)
        rng.shuffle(keys)
        total = sum(len(groups[k]) for k in keys)
        test_target = round(total * _TEST_FRACTION)
        val_target = round(total * _VAL_FRACTION)
        counts = {"test": 0, "valid": 0}
        for key in keys:
            if counts["test"] < test_target:
                split = "test"
            elif counts["valid"] < val_target:
                split = "valid"
            else:
                split = "train"
            if split != "train":
                counts[split] += len(groups[key])
            assignment[split].extend(groups[key])
    return assignment


def _contour_box(space: ET.Element) -> list[float] | None:
    points = [
        (float(p.get("x", 0)), float(p.get("y", 0)))
        for p in space.findall("contour/point")
    ]
    if len(points) < _MIN_CONTOUR_POINTS:
        return None
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _rotated_rect_box(space: ET.Element) -> list[float] | None:
    rect = space.find("rotatedRect")
    if rect is None:
        return None
    center, size, angle = rect.find("center"), rect.find("size"), rect.find("angle")
    if center is None or size is None or angle is None:
        return None
    cx, cy = float(center.get("x", 0)), float(center.get("y", 0))
    width, height = float(size.get("w", 0)), float(size.get("h", 0))
    theta = math.radians(float(angle.get("d", 0)))
    cos_t, sin_t = abs(math.cos(theta)), abs(math.sin(theta))
    half_x = (width / 2) * cos_t + (height / 2) * sin_t
    half_y = (width / 2) * sin_t + (height / 2) * cos_t
    return [cx - half_x, cy - half_y, cx + half_x, cy + half_y]


def _write_split(
    frames: list[_Frame], split_output_dir: Path, names: set[str]
) -> tuple[int, int, int]:
    """Emit one canonical COCO split; return (unlabelled, degenerate, fallback)."""
    split_output_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    unlabelled = degenerate = fallback = 0
    for image_id, (lot, jpg, xml) in enumerate(frames):
        file_name = f"{lot}_{jpg.name}"
        if file_name in names:
            raise ValueError(f"Duplicate output file name {file_name}")
        names.add(file_name)
        with Image.open(jpg) as im:
            width, height = im.size
        link_image(jpg, split_output_dir / file_name)
        images.append(
            {"id": image_id, "file_name": file_name, "width": width, "height": height}
        )
        for space in ET.parse(xml).getroot().findall("space"):  # noqa: S314  # nosec: B314
            occupied = space.get("occupied")
            if occupied not in ("0", "1"):
                unlabelled += 1
                continue
            box = _contour_box(space)
            if box is None:
                box = _rotated_rect_box(space)
                if box is not None:
                    fallback += 1
            if box is None:
                degenerate += 1
                continue
            x1, y1 = max(0.0, box[0]), max(0.0, box[1])
            x2, y2 = min(float(width), box[2]), min(float(height), box[3])
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                degenerate += 1
                continue
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(occupied),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "segmentation": [],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    payload = {
        "info": {"description": f"PKLot canonical COCO ({split_output_dir.name})"},
        "licenses": [
            {
                "id": 1,
                "name": "CC BY 4.0",
                "url": "https://creativecommons.org/licenses/by/4.0/",
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
    return unlabelled, degenerate, fallback
