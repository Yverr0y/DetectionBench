"""Tests for the SKU-110K (headerless CSV) -> canonical COCO adapter."""

import json
from pathlib import Path

from detectionbench.datasets.sku110k import SKU110KAdapter


def _write_dataset(raw_dir: Path) -> None:
    (raw_dir / "images").mkdir(parents=True)
    (raw_dir / "annotations").mkdir(parents=True)
    for name in ("train_1.jpg", "train_2.jpg", "val_1.jpg", "test_1.jpg"):
        (raw_dir / "images" / name).write_bytes(b"fake")

    # headerless CSV: file_name,x1,y1,x2,y2,class,image_width,image_height
    (raw_dir / "annotations" / "annotations_train.csv").write_text(
        "train_1.jpg,10,20,50,80,object,1000,800\n"
        "train_1.jpg,100,120,150,180,object,1000,800\n"
        "train_2.jpg,0,0,5,5,object,500,500\n"  # small box, still valid (w=h=5)
    )
    (raw_dir / "annotations" / "annotations_val.csv").write_text(
        "val_1.jpg,1,1,2,2,object,300,300\n"
    )
    (raw_dir / "annotations" / "annotations_test.csv").write_text(
        "test_1.jpg,1,1,2,2,object,300,300\n"
    )


def test_prepare_coco_maps_three_splits_single_class(tmp_path: Path) -> None:
    raw_dir = tmp_path / "SKU110K_fixed"
    out = tmp_path / "canonical"
    _write_dataset(raw_dir)

    SKU110KAdapter().prepare_coco(raw_dir, out)

    expectations = {"train": (2, 3), "valid": (1, 1), "test": (1, 1)}
    for split, (n_images, n_boxes) in expectations.items():
        payload = json.loads((out / split / "_annotations.coco.json").read_text())
        assert payload["categories"] == [
            {"id": 0, "name": "object", "supercategory": "none"}
        ]
        assert len(payload["images"]) == n_images
        assert len(payload["annotations"]) == n_boxes
        assert {a["category_id"] for a in payload["annotations"]} == {0}


def test_bbox_and_image_size_from_csv_row(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    out = tmp_path / "canonical"
    _write_dataset(raw_dir)

    SKU110KAdapter().prepare_coco(raw_dir, out)

    payload = json.loads((out / "train" / "_annotations.coco.json").read_text())
    img = next(im for im in payload["images"] if im["file_name"] == "train_1.jpg")
    assert (img["width"], img["height"]) == (1000, 800)
    boxes = sorted(
        a["bbox"] for a in payload["annotations"] if a["image_id"] == img["id"]
    )
    assert boxes == [[10.0, 20.0, 40.0, 60.0], [100.0, 120.0, 50.0, 60.0]]
