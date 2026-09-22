"""Tests for the KITTI (native label format) -> canonical COCO adapter."""

import json
from importlib import resources
from pathlib import Path

from PIL import Image

from detectionbench.datasets.kitti import KITTIAdapter

_LABEL_LINE = (
    "{type} 0.00 0 0.00 {left:.2f} {top:.2f} {right:.2f} {bottom:.2f} "
    "1.5 1.5 3.5 0 1.5 8 0.0"
)


def _write_label(
    path: Path, lines: list[tuple[str, float, float, float, float]]
) -> None:
    path.write_text(
        "\n".join(
            _LABEL_LINE.format(type=t, left=left, top=top, right=right, bottom=bottom)
            for t, left, top, right, bottom in lines
        )
    )


def _real_split_ids() -> tuple[set[str], set[str]]:
    train = (
        resources.files("detectionbench.datasets") / "data" / "kitti_train_ids.txt"
    ).read_text()
    valid = (
        resources.files("detectionbench.datasets") / "data" / "kitti_val_ids.txt"
    ).read_text()
    return (
        {ln.strip() for ln in train.splitlines() if ln.strip()},
        {ln.strip() for ln in valid.splitlines() if ln.strip()},
    )


def _make_raw(raw_dir: Path, ids: list[str]) -> None:
    image_dir = raw_dir / "training" / "image_2"
    label_dir = raw_dir / "training" / "label_2"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for stem in ids:
        Image.new("RGB", (1242, 375)).save(image_dir / f"{stem}.png")
        _write_label(
            label_dir / f"{stem}.txt",
            [
                ("Car", 10, 20, 40, 60),  # kept
                ("DontCare", 0, 0, 1242, 375),  # ignore region -> dropped
                ("Pedestrian", 100, 100, 100, 150),  # zero-width -> dropped
            ],
        )


def test_bundled_split_files_are_well_formed() -> None:
    train_ids, val_ids = _real_split_ids()
    assert len(train_ids) == 3712
    assert len(val_ids) == 3769
    assert train_ids.isdisjoint(val_ids)
    assert all(len(i) == 6 and i.isdigit() for i in train_ids | val_ids)


def test_prepare_coco_uses_bundled_split_and_drops_dontcare(tmp_path: Path) -> None:
    train_ids, val_ids = _real_split_ids()
    some_train = sorted(train_ids)[:5]
    some_val = sorted(val_ids)[:3]
    raw_dir = tmp_path / "kitti_raw"
    out = tmp_path / "canonical"
    _make_raw(raw_dir, some_train + some_val)

    KITTIAdapter().prepare_coco(raw_dir, out)

    for split, expected_ids in (("train", some_train), ("valid", some_val)):
        payload = json.loads((out / split / "_annotations.coco.json").read_text())
        assert [c["name"] for c in payload["categories"]] == KITTIAdapter.spec.classes
        # only images present on disk (the ones we wrote) show up
        got_stems = {Path(im["file_name"]).stem for im in payload["images"]}
        assert got_stems == set(expected_ids)
        # DontCare and the zero-width Pedestrian box are both dropped -> 1 box/image
        assert len(payload["annotations"]) == len(payload["images"])
        car_id = KITTIAdapter.spec.classes.index("Car")
        assert {a["category_id"] for a in payload["annotations"]} == {car_id}
        assert all(
            a["bbox"] == [10.0, 20.0, 30.0, 40.0] for a in payload["annotations"]
        )
        for im in payload["images"]:
            assert (out / split / im["file_name"]).exists()
