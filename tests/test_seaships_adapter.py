"""Tests for the SeaShips(7000) Pascal-VOC -> canonical COCO adapter."""

import json
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree

from detectionbench.datasets.seaships import SeaShipsAdapter


def _write_voc_xml(path: Path, width: int, height: int, objects: list[tuple]) -> None:
    root = Element("annotation")
    size = SubElement(root, "size")
    SubElement(size, "width").text = str(width)
    SubElement(size, "height").text = str(height)
    for name, x1, y1, x2, y2 in objects:
        obj = SubElement(root, "object")
        SubElement(obj, "name").text = name
        box = SubElement(obj, "bndbox")
        SubElement(box, "xmin").text = str(x1)
        SubElement(box, "ymin").text = str(y1)
        SubElement(box, "xmax").text = str(x2)
        SubElement(box, "ymax").text = str(y2)
    ElementTree(root).write(path)


def _write_dataset(raw_dir: Path) -> None:
    (raw_dir / "JPEGImages").mkdir(parents=True)
    (raw_dir / "Annotations").mkdir(parents=True)
    (raw_dir / "ImageSets" / "Main").mkdir(parents=True)

    stems = {"train": ["000001"], "val": ["000002"], "test": ["000003", "000004"]}
    for split, split_stems in stems.items():
        (raw_dir / "ImageSets" / "Main" / f"{split}.txt").write_text(
            "\n".join(split_stems) + "\n"
        )
    for stem in ["000001", "000002", "000003", "000004"]:
        (raw_dir / "JPEGImages" / f"{stem}.jpg").write_bytes(b"fake")
        _write_voc_xml(
            raw_dir / "Annotations" / f"{stem}.xml",
            width=1920,
            height=1080,
            objects=[
                ("ore carrier", 100, 50, 300, 150),
                ("passenger ship", 10, 10, 20, 20),
                (
                    "unknown vessel",
                    0,
                    0,
                    5,
                    5,
                ),  # outside the 6-class taxonomy -> dropped
            ],
        )


def test_prepare_coco_respects_official_split_and_six_classes(tmp_path: Path) -> None:
    raw_dir = tmp_path / "SeaShips_7000"
    out = tmp_path / "canonical"
    _write_dataset(raw_dir)

    SeaShipsAdapter().prepare_coco(raw_dir, out)

    expected_counts = {"train": 1, "valid": 1, "test": 2}
    for split, n_images in expected_counts.items():
        payload = json.loads((out / split / "_annotations.coco.json").read_text())
        assert [c["name"] for c in payload["categories"]] == [
            "ore carrier",
            "bulk cargo carrier",
            "general cargo ship",
            "container ship",
            "fishing boat",
            "passenger ship",
        ]
        assert len(payload["images"]) == n_images
        # "unknown vessel" dropped -> 2 boxes/image (ore carrier + passenger ship)
        assert len(payload["annotations"]) == 2 * n_images
        assert {a["category_id"] for a in payload["annotations"]} == {0, 5}


def test_bbox_absolute_xywh_from_voc_corners(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    out = tmp_path / "canonical"
    _write_dataset(raw_dir)

    SeaShipsAdapter().prepare_coco(raw_dir, out)

    payload = json.loads((out / "train" / "_annotations.coco.json").read_text())
    ore = next(a for a in payload["annotations"] if a["category_id"] == 0)
    assert ore["bbox"] == [100.0, 50.0, 200.0, 100.0]
