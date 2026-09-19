"""Tests for the UAVDT Supervisely -> canonical COCO adapter."""

import json
from pathlib import Path

from detectionbench.datasets.uavdt import UAVDTAdapter


def _ann(objects: list[dict]) -> dict:
    return {"size": {"height": 540, "width": 1024}, "objects": objects}


def _rect(cls: str, x1: int, y1: int, x2: int, y2: int) -> dict:
    return {
        "geometryType": "rectangle",
        "classTitle": cls,
        "points": {"exterior": [[x1, y1], [x2, y2]]},
    }


def _write_ninja(raw_dir: Path) -> None:
    for split, seqs in {"train": ["M0101", "M0202"], "test": ["S0303"]}.items():
        (raw_dir / split / "img").mkdir(parents=True)
        (raw_dir / split / "ann").mkdir(parents=True)
        for seq in seqs:
            for frame in range(4):
                name = f"{seq}_img{frame:06d}.jpg"
                (raw_dir / split / "img" / name).write_bytes(b"fake")
                objs = [
                    _rect(
                        "car", 100, 40, 20, 10
                    ),  # unordered corners -> x 20,y 10,w 80,h 30
                    _rect("bus", 5, 5, 55, 65),
                    _rect("vehicle", 0, 0, 10, 10),  # unknown class -> dropped
                ]
                (raw_dir / split / "ann" / f"{name}.json").write_text(
                    json.dumps(_ann(objs))
                )


def test_prepare_coco_three_classes_and_sequence_aware_split(tmp_path: Path) -> None:
    raw_dir = tmp_path / "uavdt-DatasetNinja"
    out = tmp_path / "canonical"
    _write_ninja(raw_dir)

    UAVDTAdapter().prepare_coco(raw_dir, out)

    seqs = {}
    total_boxes = 0
    for split in ("train", "valid", "test"):
        payload = json.loads((out / split / "_annotations.coco.json").read_text())
        assert [c["name"] for c in payload["categories"]] == ["car", "truck", "bus"]
        # "vehicle" objects are dropped -> 2 boxes per image, all category_id in {0,2}
        assert {a["category_id"] for a in payload["annotations"]} <= {0, 2}
        assert len(payload["annotations"]) == 2 * len(payload["images"])
        total_boxes += len(payload["annotations"])
        seqs[split] = {im["file_name"].split("_img")[0] for im in payload["images"]}

    # train has 2 sequences -> 1 held out for valid, disjoint
    assert seqs["train"].isdisjoint(seqs["valid"])
    assert seqs["train"] | seqs["valid"] == {"M0101", "M0202"}
    assert seqs["test"] == {"S0303"}
    assert total_boxes == 2 * (8 + 4)  # 8 train-pool images + 4 test images


def test_bbox_corners_normalized(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    out = tmp_path / "canonical"
    _write_ninja(raw_dir)

    UAVDTAdapter().prepare_coco(raw_dir, out)

    payload = json.loads((out / "test" / "_annotations.coco.json").read_text())
    car = next(a for a in payload["annotations"] if a["category_id"] == 0)
    # exterior [[100,40],[20,10]] -> x_min 20, y_min 10, w 80, h 30
    assert car["bbox"] == [20.0, 10.0, 80.0, 30.0]


def _write_test_sequences(raw_dir: Path, sequences: dict[str, list[int]]) -> None:
    """Write test frames; ``sequences`` maps a sequence id to per-frame box counts."""
    (raw_dir / "test" / "img").mkdir(parents=True, exist_ok=True)
    (raw_dir / "test" / "ann").mkdir(parents=True, exist_ok=True)
    for seq, box_counts in sequences.items():
        for frame, count in enumerate(box_counts):
            name = f"{seq}_img{frame:06d}.jpg"
            (raw_dir / "test" / "img" / name).write_bytes(b"fake")
            objs = [_rect("car", 10, 10, 50, 40) for _ in range(count)]
            (raw_dir / "test" / "ann" / f"{name}.json").write_text(
                json.dumps(_ann(objs))
            )


def test_unlabelled_sequences_are_dropped_but_partially_empty_ones_kept(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    out = tmp_path / "canonical"
    _write_ninja(raw_dir)
    # S9999: no boxes on any frame (a tracking-only sequence) -> dropped.
    # M0404: some empty frames, but labelled overall -> kept, empty frames included.
    _write_test_sequences(raw_dir, {"S9999": [0, 0, 0], "M0404": [2, 0, 1, 0]})

    UAVDTAdapter().prepare_coco(raw_dir, out)

    payload = json.loads((out / "test" / "_annotations.coco.json").read_text())
    names = {im["file_name"].split("_img")[0] for im in payload["images"]}
    assert "S9999" not in names
    assert "M0404" in names
    m0404 = [im for im in payload["images"] if im["file_name"].startswith("M0404")]
    assert len(m0404) == 4  # empty frames of a labelled sequence stay as negatives
    assert not (out / "test" / "S9999_img000000.jpg").exists()
    # Image ids stay contiguous after dropping frames.
    ids = [im["id"] for im in payload["images"]]
    assert ids == list(range(1, len(ids) + 1))
