"""Tests for the GC10-DET Supervisely -> canonical COCO adapter."""

import json
from pathlib import Path

from detectionbench.datasets.gc10det import GC10DetAdapter


def _write_ninja_export(raw_dir: Path, n_images: int = 12) -> None:
    img_dir = raw_dir / "ds" / "img"
    ann_dir = raw_dir / "ds" / "ann"
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    for i in range(n_images):
        name = f"img_{i:02d}.jpg"
        (img_dir / name).write_bytes(b"fake-image-bytes")
        ann = {
            "size": {"height": 200, "width": 400},
            "objects": [
                {
                    "geometryType": "rectangle",
                    # "waist folding" must normalize to the "waist_folding" class
                    "classTitle": "waist folding",
                    "points": {"exterior": [[300, 40], [20, 10]]},  # unordered corners
                },
                {
                    "geometryType": "rectangle",
                    "classTitle": "welding_line",
                    "points": {"exterior": [[1, 1], [399, 30]]},
                },
            ],
        }
        (ann_dir / f"{name}.json").write_text(json.dumps(ann))


def test_prepare_coco_covers_every_image_across_splits(tmp_path: Path) -> None:
    raw_dir = tmp_path / "gc10-det-DatasetNinja"
    output_dir = tmp_path / "canonical"
    _write_ninja_export(raw_dir, n_images=12)

    GC10DetAdapter().prepare_coco(raw_dir, output_dir)

    total_images = 0
    total_annotations = 0
    for split in ("train", "valid", "test"):
        payload = json.loads(
            (output_dir / split / "_annotations.coco.json").read_text()
        )
        assert [c["name"] for c in payload["categories"]] == list(
            GC10DetAdapter.spec.classes
        )
        total_images += len(payload["images"])
        total_annotations += len(payload["annotations"])

    assert total_images == 12
    assert total_annotations == 24  # two boxes per image


def test_bbox_corners_are_normalized_and_class_titles_mapped(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "canonical"
    _write_ninja_export(raw_dir, n_images=1)  # 80/10/10 of 1 -> lands in test

    GC10DetAdapter().prepare_coco(raw_dir, output_dir)

    payload = json.loads((output_dir / "test" / "_annotations.coco.json").read_text())
    names = {c["id"]: c["name"] for c in payload["categories"]}
    by_class = {names[a["category_id"]]: a for a in payload["annotations"]}

    assert "waist_folding" in by_class
    # exterior [[300,40],[20,10]] -> x_min 20, y_min 10, w 280, h 30
    assert by_class["waist_folding"]["bbox"] == [20.0, 10.0, 280.0, 30.0]
    assert (output_dir / "test" / "img_00.jpg").exists()
