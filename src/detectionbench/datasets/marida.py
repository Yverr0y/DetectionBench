"""
MARIDA (Marine Debris Archive) adapter -- marine debris detection from Sentinel-2.

Source: the official MARIDA release (Kikaki et al., PLOS ONE 2022),
redistributed via Source Cooperative (source.coop/ntua/marida) and Zenodo
(doi:10.5281/zenodo.5151941): 256x256 11-band Sentinel-2 surface-reflectance
patches (``patches/S2_<date>_<tile>/S2_<date>_<tile>_<n>.tif``) plus a
matching per-pixel classification mask (``..._cl.tif``, values 1-15, 0 =
unlabeled) and the official train/val/test patch-id lists
(``splits/{train,val,test}_X.txt``).

MARIDA is natively a **weakly-supervised semantic segmentation** dataset --
only a sparse subset of each patch's pixels are confidently labeled (large,
visually-obvious areas like cloud shadows are frequently left unlabeled),
not an exhaustively-annotated detection benchmark. This adapter converts it
to bounding-box detection via connected-component extraction per class
value in the mask (8-connectivity; components smaller than
``_MIN_COMPONENT_PIXELS`` are dropped as extraction noise) -- a box here
means "a confidently-labeled contiguous region of this class", not
"every visible instance of this class". All 15 of the original classes are
kept (marine debris, sargassum, ships, and several water/cloud "context"
classes the paper also annotates); the latter describe extended surface
phenomena rather than compact objects, so they naturally produce larger,
sparser boxes than a typical detection dataset's foreground classes.

MARIDA ships 11-band raw reflectance (Sentinel-2 bands B01-B08/B8A/B11/B12,
omitting B09/B10's atmospheric-correction-only bands), not natural RGB --
this adapter derives a true-color-ish RGB image (bands B04/B03/B02, a
per-patch percentile contrast stretch + gamma) since DetectionBench's
downstream pipeline (YOLO/RT-DETR/RF-DETR) expects standard 3-channel
images. This is a derived rendering for detection training/display, not a
radiometrically calibrated product -- the box coordinates themselves come
straight from the classification mask, independent of this rendering choice.

License: CC-BY-4.0 (confirmed on source.coop/ntua/marida and the dataset's
own README) -- attribution required, redistribution/mirroring permitted.
Imagery is Sentinel-2 (ESA Copernicus open-data programme), so no
second-order sensor-rights restriction (unlike HRSID's TerraSAR-X portions).

Stats: see docs/datasets/marida/README.md (class distribution, split
summary, box geometry -- generated via detectionbench-dataset-stats).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
from scipy import ndimage

from detectionbench.datasets.base import (
    COCO_ANNOTATION_FILENAME,
    DatasetAdapter,
    DatasetSpec,
)
from detectionbench.datasets.registry import register

_CLASSES = [
    "Marine Debris",
    "Dense Sargassum",
    "Sparse Sargassum",
    "Natural Organic Material",
    "Ship",
    "Clouds",
    "Marine Water",
    "Sediment-Laden Water",
    "Foam",
    "Turbid Water",
    "Shallow Water",
    "Waves",
    "Cloud Shadows",
    "Wakes",
    "Mixed Water",
]

# Band indices into the 11-band patch array for a true-color-ish RGB render:
# Sentinel-2 B04 (665nm, red), B03 (560nm, green), B02 (490nm, blue).
_RGB_BAND_INDICES = (3, 2, 1)
_STRETCH_LOW_PERCENTILE = 1.0
_STRETCH_HIGH_PERCENTILE = 99.0
_GAMMA = 1.5

_MIN_COMPONENT_PIXELS = 4
_CONNECTIVITY = np.ones((3, 3))  # 8-connectivity for connected-component labeling

# Official split -> canonical split name + the patch-id list file that defines it.
_SPLIT_FILES = {"train": "train_X.txt", "valid": "val_X.txt", "test": "test_X.txt"}


@register
class MARIDAAdapter(DatasetAdapter):
    """Adapter for the MARIDA Sentinel-2 marine-debris dataset."""

    spec = DatasetSpec(
        key="marida",
        display_name="MARIDA",
        classes=_CLASSES,
        description=(
            "MARIDA (Marine Debris Archive) is a Sentinel-2 satellite-imagery dataset "
            "for marine debris and related ocean-surface phenomena: 1,381 256x256 "
            "patches (11-band reflectance) with per-pixel classification masks across "
            "15 classes -- marine debris, sargassum, ships, foam, and several water/ "
            "cloud context classes. It's used to benchmark marine-debris monitoring "
            "from freely-available satellite data; this adapter converts its native "
            "weakly-supervised segmentation masks into bounding boxes via "
            "connected-component extraction."
        ),
        homepage="https://marine-debris.github.io/",
        github="https://github.com/marine-debris/marine-debris.github.io",
        citation=(
            "@article{kikaki2022marida,\n"
            "  title={MARIDA: A benchmark for Marine Debris detection from Sentinel-2 "
            "remote sensing data},\n"
            "  author={Kikaki, Katerina and Kakogeorgiou, Ioannis and Mikeli, "
            "Paraskevi and Raitsos, Dionysios E. and Karantzalos, Konstantinos},\n"
            "  journal={PLOS ONE},\n"
            "  volume={17},\n"
            "  number={1},\n"
            "  pages={e0262247},\n"
            "  year={2022}\n"
            "}"
        ),
        license="CC-BY-4.0 -- attribution required; redistribution permitted.",
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert a raw MARIDA download into the canonical COCO layout."""
        splits_dir = raw_dir / "splits"
        patches_dir = raw_dir / "patches"
        if not splits_dir.is_dir() or not patches_dir.is_dir():
            raise FileNotFoundError(f"Expected {raw_dir}/splits and {raw_dir}/patches.")
        output_dir.mkdir(parents=True, exist_ok=True)

        for target_split, split_file in _SPLIT_FILES.items():
            split_path = splits_dir / split_file
            if not split_path.exists():
                continue
            roi_ids = [
                line.strip()
                for line in split_path.read_text().splitlines()
                if line.strip()
            ]
            _convert_split(roi_ids, patches_dir, output_dir / target_split)


def _roi_folder(roi_id: str) -> str:
    """S2_<date>_<tile> scene-folder name from a '<date>_<tile>_<n>' patch id."""
    return "S2_" + "_".join(roi_id.split("_")[:-1])


def _render_rgb(bands: np.ndarray) -> np.ndarray:
    """Derive an 8-bit true-color-ish RGB render from an (H,W,11) reflectance array."""
    rgb = bands[:, :, _RGB_BAND_INDICES].copy()
    # A handful of scene-edge/no-data pixels come through as NaN (same as the
    # official dataloader observes) -- impute with this patch's own per-channel
    # mean rather than let them propagate into the stretch/uint8 cast as garbage.
    nan_mask = np.isnan(rgb)
    if nan_mask.any():
        with np.errstate(invalid="ignore"):
            channel_means = np.nanmean(rgb, axis=(0, 1))
        channel_means = np.nan_to_num(channel_means, nan=0.0)  # an all-NaN channel
        for channel in range(rgb.shape[2]):
            rgb[nan_mask[:, :, channel], channel] = channel_means[channel]

    low = np.percentile(rgb, _STRETCH_LOW_PERCENTILE)
    high = np.percentile(rgb, _STRETCH_HIGH_PERCENTILE)
    stretched = np.clip((rgb - low) / max(high - low, 1e-6), 0.0, 1.0)
    stretched = np.power(stretched, 1.0 / _GAMMA)
    return (stretched * 255).astype(np.uint8)


def _extract_boxes(mask: np.ndarray) -> list[tuple[int, float, float, float, float]]:
    """Connected-component bbox extraction per class value (1-15; 0 = unlabeled)."""
    boxes: list[tuple[int, float, float, float, float]] = []
    for class_value in sorted(int(v) for v in np.unique(mask) if v > 0):
        binary = (mask == class_value).astype(np.uint8)
        labeled, n_components = ndimage.label(binary, structure=_CONNECTIVITY)
        for component_id in range(1, n_components + 1):
            ys, xs = np.where(labeled == component_id)
            if len(ys) < _MIN_COMPONENT_PIXELS:
                continue
            x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
            boxes.append(
                (
                    class_value - 1,
                    float(x1),
                    float(y1),
                    float(x2 - x1 + 1),
                    float(y2 - y1 + 1),
                )
            )
    return boxes


def _convert_split(
    roi_ids: list[str], patches_dir: Path, split_output_dir: Path
) -> None:
    """Render each patch to RGB + extract boxes from its mask into one COCO split."""
    split_output_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1

    for image_id, roi_id in enumerate(roi_ids, start=1):
        folder = _roi_folder(roi_id)
        image_path = patches_dir / folder / f"S2_{roi_id}.tif"
        mask_path = patches_dir / folder / f"S2_{roi_id}_cl.tif"
        if not image_path.exists() or not mask_path.exists():
            continue

        bands = tifffile.imread(image_path)
        mask = tifffile.imread(mask_path).astype(np.int32)
        height, width = mask.shape

        rgb = _render_rgb(bands)
        file_name = f"{roi_id}.jpg"
        cv2.imwrite(
            str(split_output_dir / file_name),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        images.append(
            {"id": image_id, "file_name": file_name, "width": width, "height": height}
        )

        for category_id, x, y, box_width, box_height in _extract_boxes(mask):
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, box_width, box_height],
                    "area": box_width * box_height,
                    "segmentation": [],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    payload = {
        "info": {"description": f"MARIDA canonical COCO ({split_output_dir.name})"},
        "licenses": [
            {
                "id": 1,
                "name": "CC-BY-4.0",
                "url": "https://creativecommons.org/licenses/by/4.0/",
            }
        ],
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
