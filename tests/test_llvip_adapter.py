"""Tests for the LLVIP (Pascal VOC, infrared-only) -> canonical COCO adapter."""

import json
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement

from detectionbench.datasets.llvip import LLVIPAdapter


def _write_voc_xml(path: Path, width: int, height: int, boxes: list[tuple]) -> None:
    root = Element("annotation")
    size = SubElement(root, "size")
    SubElement(size, "width").text = str(width)
    SubElement(size, "height").text = str(height)
    for x1, y1, x2, y2 in boxes:
        obj = SubElement(root, "object")
        SubElement(obj, "name").text = "person"
        box = SubElement(obj, "bndbox")
        SubElement(box, "xmin").text = str(x1)
        SubElement(box, "ymin").text = str(y1)
        SubElement(box, "xmax").text = str(x2)
        SubElement(box, "ymax").text = str(y2)
    ElementTree(root).write(path)


def _write_dataset(raw_dir: Path) -> None:
    (raw_dir / "infrared" / "train").mkdir(parents=True)
    (raw_dir / "infrared" / "test").mkdir(parents=True)
    (raw_dir / "visible" / "train").mkdir(parents=True)  # present but unused
    (raw_dir / "Annotations").mkdir(parents=True)

    for stem in [f"{i:06d}" for i in range(1, 21)] + ["900001"]:
        split = "test" if stem == "900001" else "train"
        (raw_dir / "infrared" / split / f"{stem}.jpg").write_bytes(b"fake")
        _write_voc_xml(
            raw_dir / "Annotations" / f"{stem}.xml",
            width=1280,
            height=1024,
            boxes=[(100, 50, 300, 400)],
        )


def test_prepare_coco_uses_infrared_and_splits_train(tmp_path: Path) -> None:
    raw_dir = tmp_path / "LLVIP"
    out = tmp_path / "canonical"
    _write_dataset(raw_dir)

    LLVIPAdapter().prepare_coco(raw_dir, out)

    counts = {}
    for split in ("train", "valid", "test"):
        payload = json.loads((out / split / "_annotations.coco.json").read_text())
        assert payload["categories"] == [
            {"id": 0, "name": "person", "supercategory": "none"}
        ]
        assert len(payload["annotations"]) == len(payload["images"])  # one box each
        counts[split] = len(payload["images"])

    assert counts["train"] + counts["valid"] == 20  # 20 train frames
    assert counts["valid"] == 3  # round(20 * 0.15)
    assert counts["test"] == 1


def test_bbox_absolute_xywh_from_voc_corners(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    out = tmp_path / "canonical"
    _write_dataset(raw_dir)

    LLVIPAdapter().prepare_coco(raw_dir, out)

    payload = json.loads((out / "test" / "_annotations.coco.json").read_text())
    assert payload["annotations"][0]["bbox"] == [100.0, 50.0, 200.0, 350.0]
