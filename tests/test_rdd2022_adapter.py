"""Tests for the RDD2022 (RDD_SPLIT YOLO tree -> canonical COCO) adapter."""

import json
from pathlib import Path

from PIL import Image

from detectionbench.datasets.rdd2022 import RDD2022Adapter


def _write_rdd_split(raw_dir: Path) -> None:
    for split in ("train", "val", "test"):
        (raw_dir / split / "images").mkdir(parents=True)
        (raw_dir / split / "labels").mkdir(parents=True)
        img_path = raw_dir / split / "images" / "Japan_000001.jpg"
        Image.new("RGB", (200, 100), color="white").save(img_path)
        # 3 valid boxes (classes 0,1,3) + one class-4 line that must be dropped
        (raw_dir / split / "labels" / "Japan_000001.txt").write_text(
            "0 0.5 0.5 0.2 0.4\n"
            "1 0.25 0.25 0.1 0.1\n"
            "3 0.75 0.75 0.3 0.2\n"
            "4 0.5 0.5 0.5 0.5\n"
        )


def test_prepare_coco_maps_four_classes_and_drops_class_four(tmp_path: Path) -> None:
    raw_dir = tmp_path / "RDD_SPLIT"
    output_dir = tmp_path / "canonical"
    _write_rdd_split(raw_dir)

    RDD2022Adapter().prepare_coco(raw_dir, output_dir)

    # val -> valid
    assert (output_dir / "valid" / "_annotations.coco.json").exists()

    for split in ("train", "valid", "test"):
        payload = json.loads(
            (output_dir / split / "_annotations.coco.json").read_text()
        )
        assert [c["name"] for c in payload["categories"]] == [
            "longitudinal_crack",
            "transverse_crack",
            "alligator_crack",
            "pothole",
        ]
        assert len(payload["images"]) == 1
        # class-4 line dropped -> 3 boxes, all category_id in 0..3
        assert len(payload["annotations"]) == 3
        assert {a["category_id"] for a in payload["annotations"]} == {0, 1, 3}
        assert (output_dir / split / "Japan_000001.jpg").exists()


def test_bbox_denormalized_to_absolute_xywh(tmp_path: Path) -> None:
    raw_dir = tmp_path / "RDD_SPLIT"
    output_dir = tmp_path / "canonical"
    _write_rdd_split(raw_dir)

    RDD2022Adapter().prepare_coco(raw_dir, output_dir)

    payload = json.loads((output_dir / "train" / "_annotations.coco.json").read_text())
    first = next(a for a in payload["annotations"] if a["category_id"] == 0)
    # 0 0.5 0.5 0.2 0.4 on a 200x100 image -> x 80, y 30, w 40, h 40
    assert first["bbox"] == [80.0, 30.0, 40.0, 40.0]
