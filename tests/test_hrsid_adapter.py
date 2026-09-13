"""Tests for the HRSID (COCO) -> canonical COCO adapter."""

import json
from pathlib import Path

from detectionbench.datasets.hrsid import HRSIDAdapter


def _coco(image_ids: list[int]) -> dict:
    images = [
        {"id": i, "file_name": f"P{i:04d}_0_800_0_800.jpg", "width": 800, "height": 800}
        for i in image_ids
    ]
    annotations = [
        {
            "id": 1000 + i,
            "image_id": i,
            "category_id": 1,  # HRSID's single class id -> must be remapped to 0
            "bbox": [10.0, 20.0, 30.0, 40.0],
            "area": 1200.0,
            "segmentation": [[1, 2, 3, 4, 5, 6]],
            "iscrowd": 0,
        }
        for i in image_ids
    ]
    return {
        "info": {},
        "licenses": [],
        "images": images,
        "type": "instances",
        "annotations": annotations,
        "categories": [{"supercategory": None, "id": 1, "name": "ship"}],
    }


def _write_hrsid(raw_dir: Path) -> None:
    (raw_dir / "JPEGImages").mkdir(parents=True)
    (raw_dir / "annotations").mkdir(parents=True)
    train_ids = list(range(20))
    test_ids = list(range(100, 106))
    for i in train_ids + test_ids:
        (raw_dir / "JPEGImages" / f"P{i:04d}_0_800_0_800.jpg").write_bytes(b"fake")
    (raw_dir / "annotations" / "train2017.json").write_text(
        json.dumps(_coco(train_ids))
    )
    (raw_dir / "annotations" / "test2017.json").write_text(json.dumps(_coco(test_ids)))


def test_prepare_coco_splits_and_remaps_single_class(tmp_path: Path) -> None:
    raw_dir = tmp_path / "HRSID_JPG"
    out = tmp_path / "canonical"
    _write_hrsid(raw_dir)

    HRSIDAdapter().prepare_coco(raw_dir, out)

    counts = {}
    for split in ("train", "valid", "test"):
        payload = json.loads((out / split / "_annotations.coco.json").read_text())
        assert payload["categories"] == [
            {"id": 0, "name": "ship", "supercategory": "none"}
        ]
        assert {a["category_id"] for a in payload["annotations"]} == {0}
        # segmentation stripped, one box per image, absolute xywh preserved
        assert all(a["segmentation"] == [] for a in payload["annotations"])
        assert all(
            a["bbox"] == [10.0, 20.0, 30.0, 40.0] for a in payload["annotations"]
        )
        counts[split] = len(payload["images"])
        assert (out / split / payload["images"][0]["file_name"]).exists()

    # train2017 (20) -> train + valid, 15% held out ; test2017 (6) -> test
    assert counts["train"] + counts["valid"] == 20
    assert counts["valid"] == 3  # round(20 * 0.15)
    assert counts["test"] == 6
