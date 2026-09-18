"""Tests for the MARIDA (multi-band GeoTIFF + mask) -> canonical COCO adapter."""

import json
from pathlib import Path

import numpy as np
import tifffile

from detectionbench.datasets.marida import MARIDAAdapter

_PATCH_SIZE = 32


def _write_patch(patches_dir: Path, roi_id: str, mask: np.ndarray) -> None:
    folder = "S2_" + "_".join(roi_id.split("_")[:-1])
    patch_dir = patches_dir / folder
    patch_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    bands = rng.uniform(0.02, 0.2, size=(_PATCH_SIZE, _PATCH_SIZE, 11)).astype(
        np.float32
    )
    tifffile.imwrite(patch_dir / f"S2_{roi_id}.tif", bands)
    tifffile.imwrite(patch_dir / f"S2_{roi_id}_cl.tif", mask.astype(np.float32))


def _write_dataset(raw_dir: Path) -> None:
    splits_dir = raw_dir / "splits"
    splits_dir.mkdir(parents=True)

    # A 6x6 block of class 5 (Ship, id 4 after 0-index) at a known location,
    # plus a 2-pixel noise speck of class 1 (Marine Debris) that should be
    # dropped by the minimum-component-size filter.
    train_mask = np.zeros((_PATCH_SIZE, _PATCH_SIZE), dtype=np.int32)
    train_mask[4:10, 8:14] = 5  # rows 4-9, cols 8-13 -> bbox x=8,y=4,w=6,h=6
    train_mask[20:21, 20:22] = 1  # 2 pixels -> below _MIN_COMPONENT_PIXELS (4)
    _write_patch(raw_dir / "patches", "1-1-20_10ABC_0", train_mask)
    (splits_dir / "train_X.txt").write_text("1-1-20_10ABC_0\n")

    # Two disjoint blobs of the same class -> two separate boxes.
    val_mask = np.zeros((_PATCH_SIZE, _PATCH_SIZE), dtype=np.int32)
    val_mask[0:3, 0:3] = 2  # Dense Sargassum
    val_mask[25:29, 25:29] = 2
    _write_patch(raw_dir / "patches", "2-2-20_20XYZ_0", val_mask)
    (splits_dir / "val_X.txt").write_text("2-2-20_20XYZ_0\n")

    empty_mask = np.zeros((_PATCH_SIZE, _PATCH_SIZE), dtype=np.int32)
    _write_patch(raw_dir / "patches", "3-3-20_30QRS_0", empty_mask)
    (splits_dir / "test_X.txt").write_text("3-3-20_30QRS_0\n")


def test_prepare_coco_extracts_boxes_and_renders_rgb(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    out = tmp_path / "canonical"
    _write_dataset(raw_dir)

    MARIDAAdapter().prepare_coco(raw_dir, out)

    train_payload = json.loads((out / "train" / "_annotations.coco.json").read_text())
    assert len(train_payload["images"]) == 1
    # only the 6x6 Ship block survives the min-component-size filter
    assert len(train_payload["annotations"]) == 1
    ann = train_payload["annotations"][0]
    assert ann["category_id"] == 4  # Ship is class id 5, 0-indexed -> 4
    assert ann["bbox"] == [8.0, 4.0, 6.0, 6.0]

    rendered = out / "train" / "1-1-20_10ABC_0.jpg"
    assert rendered.exists()

    valid_payload = json.loads((out / "valid" / "_annotations.coco.json").read_text())
    assert len(valid_payload["annotations"]) == 2  # two disjoint Sargassum blobs
    assert {a["category_id"] for a in valid_payload["annotations"]} == {1}

    test_payload = json.loads((out / "test" / "_annotations.coco.json").read_text())
    assert (
        test_payload["annotations"] == []
    )  # empty mask -> no boxes, still a valid image
    assert len(test_payload["images"]) == 1

    for split in ("train", "valid", "test"):
        payload = json.loads((out / split / "_annotations.coco.json").read_text())
        assert len(payload["categories"]) == 15
        assert payload["categories"][0]["name"] == "Marine Debris"
        assert payload["categories"][4]["name"] == "Ship"


def test_roi_folder_parsing_matches_official_scheme() -> None:
    from detectionbench.datasets.marida import _roi_folder

    assert _roi_folder("1-12-19_48MYU_0") == "S2_1-12-19_48MYU"
    assert _roi_folder("11-6-18_16PCC_23") == "S2_11-6-18_16PCC"
