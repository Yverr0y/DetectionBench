"""Tests for the DUO (COCO) -> canonical COCO adapter."""

import json
from pathlib import Path

from detectionbench.datasets.duo import DUOAdapter


def _coco(image_ids: list[int]) -> dict:
    images = [
        {"id": i, "file_name": f"{i:06d}.jpg", "width": 480, "height": 270}
        for i in image_ids
    ]
    annotations = [
        {
            "id": 1000 + i,
            "image_id": i,
            "category_id": 2,  # DUO's "echinus" -> must remap to 0-indexed id 1
            "bbox": [5.0, 6.0, 20.0, 30.0],
            "area": 600.0,
            "iscrowd": 0,
        }
        for i in image_ids
    ]
    return {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"name": "holothurian", "id": 1},
            {"name": "echinus", "id": 2},
            {"name": "scallop", "id": 3},
            {"name": "starfish", "id": 4},
        ],
    }


def _write_duo(raw_dir: Path) -> None:
    (raw_dir / "images" / "train").mkdir(parents=True)
    (raw_dir / "images" / "test").mkdir(parents=True)
    (raw_dir / "annotations").mkdir(parents=True)
    train_ids = list(range(20))
    test_ids = list(range(100, 106))
    for i in train_ids:
        (raw_dir / "images" / "train" / f"{i:06d}.jpg").write_bytes(b"fake")
    for i in test_ids:
        (raw_dir / "images" / "test" / f"{i:06d}.jpg").write_bytes(b"fake")
    (raw_dir / "annotations" / "instances_train.json").write_text(
        json.dumps(_coco(train_ids))
    )
    (raw_dir / "annotations" / "instances_test.json").write_text(
        json.dumps(_coco(test_ids))
    )


def test_prepare_coco_splits_and_remaps_categories(tmp_path: Path) -> None:
    raw_dir = tmp_path / "DUO"
    out = tmp_path / "canonical"
    _write_duo(raw_dir)

    DUOAdapter().prepare_coco(raw_dir, out)

    counts = {}
    for split in ("train", "valid", "test"):
        payload = json.loads((out / split / "_annotations.coco.json").read_text())
        assert [c["name"] for c in payload["categories"]] == [
            "holothurian",
            "echinus",
            "scallop",
            "starfish",
        ]
        # source category_id 2 ("echinus") -> canonical id 1
        assert {a["category_id"] for a in payload["annotations"]} == {1}
        assert all(a["bbox"] == [5.0, 6.0, 20.0, 30.0] for a in payload["annotations"])
        counts[split] = len(payload["images"])
        assert (out / split / payload["images"][0]["file_name"]).exists()

    # train (20) -> train + valid, 15% held out ; test (6) -> test
    assert counts["train"] + counts["valid"] == 20
    assert counts["valid"] == 3  # round(20 * 0.15)
    assert counts["test"] == 6
