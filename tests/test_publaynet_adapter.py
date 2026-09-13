"""Tests for the PubLayNet -> canonical COCO adapter."""

import json
from pathlib import Path

from detectionbench.datasets.publaynet import PubLayNetAdapter


def _write_split(raw_dir: Path, split: str, category_ids: list[int]) -> None:
    (raw_dir / split).mkdir(parents=True)
    (raw_dir / split / "page_1.jpg").write_bytes(b"fake-image-bytes")
    payload = {
        "categories": [
            {"id": cid, "name": name, "supercategory": "none"}
            for cid, name in zip(category_ids, ["title", "text"], strict=True)
        ],
        "images": [{"id": 1, "file_name": "page_1.jpg", "width": 612, "height": 792}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": category_ids[0],
                "bbox": [0, 0, 10, 10],
            },
        ],
    }
    (raw_dir / f"{split}.json").write_text(json.dumps(payload))


def test_prepare_coco_writes_canonical_train_and_valid(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "canonical"
    # source category ids are non-contiguous (5, 2), like real PubLayNet (1-5)
    _write_split(raw_dir, "train", [5, 2])

    PubLayNetAdapter().prepare_coco(raw_dir, output_dir)

    annotation_path = output_dir / "train" / "_annotations.coco.json"
    assert annotation_path.exists()
    payload = json.loads(annotation_path.read_text())
    assert len(payload["images"]) == 1
    assert (output_dir / "train" / "page_1.jpg").exists()

    # remapped to a contiguous 0-indexed range, ordered by original id (2 -> 0, 5 -> 1)
    categories_by_name = {c["name"]: c["id"] for c in payload["categories"]}
    assert categories_by_name == {"text": 0, "title": 1}
    assert payload["annotations"][0]["category_id"] == categories_by_name["title"]

    # no test.json -> no test split, and val wasn't provided either
    assert not (output_dir / "valid").exists()
    assert not (output_dir / "test").exists()


def test_prepare_coco_val_maps_to_valid_split(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "canonical"
    _write_split(raw_dir, "train", [1, 2])
    _write_split(raw_dir, "val", [1, 2])

    PubLayNetAdapter().prepare_coco(raw_dir, output_dir)

    assert (output_dir / "valid" / "_annotations.coco.json").exists()
