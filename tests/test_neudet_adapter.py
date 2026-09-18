"""Tests for the NEU-DET (Pascal VOC) -> canonical COCO adapter."""

import json
from pathlib import Path

from detectionbench.datasets.neudet import NEUDETAdapter

_VOC_TEMPLATE = """<annotation>
  <filename>{name}.jpg</filename>
  <size>
    <width>200</width>
    <height>200</height>
    <depth>1</depth>
  </size>
  <object>
    <name>{cls}</name>
    <bndbox>
      <xmin>{x1}</xmin>
      <ymin>{y1}</ymin>
      <xmax>{x2}</xmax>
      <ymax>{y2}</ymax>
    </bndbox>
  </object>
</annotation>
"""


def _write_dataset(raw_dir: Path, n_images: int = 12) -> None:
    image_dir = raw_dir / "IMAGES"
    annotation_dir = raw_dir / "ANNOTATIONS"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)

    classes = [
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches",
    ]
    for i in range(n_images):
        name = f"{classes[i % len(classes)]}_{i}"
        (image_dir / f"{name}.jpg").write_bytes(b"fake-image-bytes")
        (annotation_dir / f"{name}.xml").write_text(
            _VOC_TEMPLATE.format(
                name=name, cls=classes[i % len(classes)], x1=4, y1=74, x2=192, y2=166
            )
        )


def test_prepare_coco_covers_every_image_across_splits(tmp_path: Path) -> None:
    raw_dir = tmp_path / "NEU-DET"
    output_dir = tmp_path / "canonical"
    _write_dataset(raw_dir, n_images=20)

    NEUDETAdapter().prepare_coco(raw_dir, output_dir)

    total_images = 0
    total_annotations = 0
    for split in ("train", "valid", "test"):
        payload = json.loads(
            (output_dir / split / "_annotations.coco.json").read_text()
        )
        assert [c["name"] for c in payload["categories"]] == list(
            NEUDETAdapter.spec.classes
        )
        total_images += len(payload["images"])
        total_annotations += len(payload["annotations"])

    assert total_images == 20
    assert total_annotations == 20  # one box per image


def test_bbox_and_class_mapping(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "canonical"
    _write_dataset(raw_dir, n_images=1)  # 80/10/10 of 1 -> lands in test

    NEUDETAdapter().prepare_coco(raw_dir, output_dir)

    payload = json.loads((output_dir / "test" / "_annotations.coco.json").read_text())
    names = {c["id"]: c["name"] for c in payload["categories"]}
    ann = payload["annotations"][0]

    assert names[ann["category_id"]] == "crazing"
    assert ann["bbox"] == [4.0, 74.0, 188.0, 92.0]
    assert (output_dir / "test" / "crazing_0.jpg").exists()
