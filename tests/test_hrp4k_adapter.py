"""Tests for the HRP4K (COCO) -> canonical COCO adapter."""

import json
from pathlib import Path

from detectionbench.datasets.hrp4k import HRP4KAdapter


def _coco(image_ids: list[int], *, missing: frozenset[int] = frozenset()) -> dict:
    images = [
        {"id": i, "file_name": f"{i}.jpg", "width": 3840, "height": 2160}
        for i in image_ids
        if i not in missing
    ]
    annotations = [
        {
            "id": 1000 + i,
            "image_id": i,
            "category_id": 0,
            "bbox": [10.0, 20.0, 30.0, 40.0],
            "area": 1200.0,
            "segmentation": [],
            "iscrowd": 0,
        }
        for i in image_ids
    ]
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"supercategory": "pothole", "id": 0, "name": "pothole"}],
    }


def _write_hrp4k(raw_dir: Path) -> None:
    for split in ("train", "valid", "test"):
        (raw_dir / split / "images").mkdir(parents=True)
        (raw_dir / split / "labels").mkdir(parents=True)

    # train.json references 20 images (0-19) but only 12 (0-11) actually
    # exist on disk -- mirrors the real HRP4K upstream completeness gap.
    train_ids = list(range(20))
    missing = frozenset(range(12, 20))
    for i in train_ids:
        if i in missing:
            continue
        (raw_dir / "train" / "images" / f"{i}.jpg").write_bytes(b"fake")
    (raw_dir / "train.json").write_text(json.dumps(_coco(train_ids, missing=missing)))

    for split, ids in (("valid", range(100, 106)), ("test", range(200, 205))):
        for i in ids:
            (raw_dir / split / "images" / f"{i}.jpg").write_bytes(b"fake")
        (raw_dir / f"{split}.json").write_text(json.dumps(_coco(list(ids))))


def test_prepare_coco_splits_single_class(tmp_path: Path) -> None:
    raw_dir = tmp_path / "HRP4K"
    out = tmp_path / "canonical"
    _write_hrp4k(raw_dir)

    HRP4KAdapter().prepare_coco(raw_dir, out)

    counts = {}
    for split in ("train", "valid", "test"):
        payload = json.loads((out / split / "_annotations.coco.json").read_text())
        assert payload["categories"] == [
            {"id": 0, "name": "pothole", "supercategory": "none"}
        ]
        assert {a["category_id"] for a in payload["annotations"]} == {0}
        assert all(
            a["bbox"] == [10.0, 20.0, 30.0, 40.0] for a in payload["annotations"]
        )
        counts[split] = len(payload["images"])
        for im in payload["images"]:
            assert (out / split / im["file_name"]).exists()
        assert len(payload["annotations"]) == len(payload["images"])

    # 20 referenced train images, only 12 exist on disk -> orphan images and
    # their annotations are dropped, not carried through as broken links.
    assert counts["train"] == 12
    assert counts["valid"] == 6
    assert counts["test"] == 5
