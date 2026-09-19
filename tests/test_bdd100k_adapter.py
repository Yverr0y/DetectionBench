"""Tests for the BDD100K (native JSON) -> canonical COCO adapter."""

import json
from pathlib import Path

from detectionbench.datasets.bdd100k import BDD100KAdapter


def _entry(name: str, categories: list[str]) -> dict:
    return {
        "name": name,
        "attributes": {
            "weather": "clear",
            "scene": "city street",
            "timeofday": "daytime",
        },
        "labels": [
            {
                "category": cat,
                "attributes": {"occluded": False, "truncated": False},
                "box2d": {"x1": 10.0, "y1": 20.0, "x2": 40.0, "y2": 60.0},
                "id": i,
            }
            for i, cat in enumerate(categories)
        ]
        # a lane/drivable-area style entry with no box2d, which must be skipped
        + [{"category": "lane", "attributes": {}, "poly2d": [], "id": 99}],
    }


def _write_bdd100k(raw_dir: Path) -> None:
    (raw_dir / "images" / "100k" / "train" / "trainA").mkdir(parents=True)
    (raw_dir / "images" / "100k" / "train" / "trainB").mkdir(parents=True)
    (raw_dir / "images" / "100k" / "val").mkdir(parents=True)
    (raw_dir / "labels").mkdir(parents=True)

    train_entries = []
    for i in range(20):
        name = f"train_{i:03d}.jpg"
        sub = "trainA" if i % 2 == 0 else "trainB"
        (raw_dir / "images" / "100k" / "train" / sub / name).write_bytes(b"fake")
        train_entries.append(_entry(name, ["car", "person"]))
    (raw_dir / "labels" / "bdd100k_labels_images_train.json").write_text(
        json.dumps(train_entries)
    )

    val_entries = []
    for i in range(6):
        name = f"val_{i:03d}.jpg"
        (raw_dir / "images" / "100k" / "val" / name).write_bytes(b"fake")
        val_entries.append(_entry(name, ["traffic light"]))
    (raw_dir / "labels" / "bdd100k_labels_images_val.json").write_text(
        json.dumps(val_entries)
    )


def test_prepare_coco_splits_multiclass(tmp_path: Path) -> None:
    raw_dir = tmp_path / "bdd100k"
    out = tmp_path / "canonical"
    _write_bdd100k(raw_dir)

    BDD100KAdapter().prepare_coco(raw_dir, out)

    counts = {}
    for split in ("train", "valid", "test"):
        payload = json.loads((out / split / "_annotations.coco.json").read_text())
        assert len(payload["categories"]) == 10
        assert all(a["segmentation"] == [] for a in payload["annotations"])
        counts[split] = len(payload["images"])
        for im in payload["images"]:
            assert (out / split / im["file_name"]).exists()
            assert im["width"] == 1280
            assert im["height"] == 720

    # 20 train entries -> train + valid, 15% held out; 6 val entries -> test.
    assert counts["train"] + counts["valid"] == 20
    assert counts["valid"] == 3  # round(20 * 0.15)
    assert counts["test"] == 6

    # 2 box2d labels/image (lane poly2d dropped) -> train+valid boxes == 40.
    train_payload = json.loads((out / "train" / "_annotations.coco.json").read_text())
    valid_payload = json.loads((out / "valid" / "_annotations.coco.json").read_text())
    assert len(train_payload["annotations"]) + len(valid_payload["annotations"]) == 40

    test_payload = json.loads((out / "test" / "_annotations.coco.json").read_text())
    assert len(test_payload["annotations"]) == 6
    assert {a["category_id"] for a in test_payload["annotations"]} == {
        BDD100KAdapter.spec.classes.index("traffic light")
    }
