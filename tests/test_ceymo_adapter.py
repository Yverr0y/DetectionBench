"""Tests for the CeyMo (VOC XML) -> canonical COCO adapter."""

import json
from pathlib import Path

from detectionbench.datasets.ceymo import CeyMoAdapter

_XML = """<annotation>
  <filename>{name}.jpg</filename>
  <size><width>1920</width><height>1080</height><depth>3</depth></size>
  {objects}
</annotation>"""

_OBJ = """<object><name>{code}</name><bndbox>
  <xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax>
</bndbox></object>"""


def _write_split(base: Path, names: list[str]) -> None:
    (base / "images").mkdir(parents=True)
    (base / "bbox_annotations").mkdir(parents=True)
    for name in names:
        (base / "images" / f"{name}.jpg").write_bytes(b"fake")
        objects = _OBJ.format(code="SA", x1=10, y1=20, x2=40, y2=60) + _OBJ.format(
            code="PC",
            x1=100,
            y1=200,
            x2=100,
            y2=260,  # zero-width -> dropped
        )
        (base / "bbox_annotations" / f"{name}.xml").write_text(
            _XML.format(name=name, objects=objects)
        )


def test_prepare_coco_splits_and_classes(tmp_path: Path) -> None:
    raw_dir = tmp_path / "CeyMo"
    out = tmp_path / "canonical"
    _write_split(raw_dir / "train", [f"t{i:03d}" for i in range(20)])
    _write_split(raw_dir / "test", [f"e{i:03d}" for i in range(6)])

    CeyMoAdapter().prepare_coco(raw_dir, out)

    counts = {}
    for split in ("train", "valid", "test"):
        payload = json.loads((out / split / "_annotations.coco.json").read_text())
        assert len(payload["categories"]) == 11
        # only the valid SA box survives; the zero-width PC box is dropped
        sa_id = CeyMoAdapter.spec.classes.index("straight_arrow")
        assert {a["category_id"] for a in payload["annotations"]} == {sa_id}
        assert all(
            a["bbox"] == [10.0, 20.0, 30.0, 40.0] for a in payload["annotations"]
        )
        assert len(payload["annotations"]) == len(payload["images"])
        counts[split] = len(payload["images"])
        for im in payload["images"]:
            assert (out / split / im["file_name"]).exists()

    assert counts["train"] + counts["valid"] == 20
    assert counts["valid"] == 3  # round(20 * 0.15)
    assert counts["test"] == 6
