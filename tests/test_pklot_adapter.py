"""Tests for the PKLot (XML) -> canonical COCO adapter."""

import json
from pathlib import Path

from PIL import Image

from detectionbench.datasets.pklot import PKLotAdapter

_RECT_ONLY = (
    '<space id="{i}" occupied="0"><rotatedRect><center x="50" y="30"/>'
    '<size w="20" h="10"/><angle d="0"/></rotatedRect></space>'
)
_SPACE = (
    '<space id="{i}"{occ}><contour>'
    '<point x="{x1}" y="{y1}"/><point x="{x2}" y="{y1}"/>'
    '<point x="{x2}" y="{y2}"/><point x="{x1}" y="{y2}"/>'
    "</contour></space>"
)


def _xml(spaces: list[tuple[str, int, int, int, int]]) -> str:
    body = "".join(
        _SPACE.format(i=i, occ=occ, x1=x1, y1=y1, x2=x2, y2=y2)
        for i, (occ, x1, y1, x2, y2) in enumerate(spaces)
    )
    return f'<parking id="t">{body}{_RECT_ONLY.format(i=99)}</parking>'


def _make_raw(raw_dir: Path) -> None:
    lots = {"LOTA": 10, "LOTB": 10}  # dates per lot, 2 frames per date
    for lot, n_dates in lots.items():
        for d in range(n_dates):
            date = f"2012-09-{d + 1:02d}"
            # folder date deliberately differs from the filename date for one
            # group, and weather folders differ -- grouping must use the filename
            folder = f"{lot}/Sunny/2011-01-01" if d == 0 else f"{lot}/Rainy/{date}"
            (raw_dir / "PKLot" / folder).mkdir(parents=True, exist_ok=True)
            for t in ("08_00_00", "09_00_00"):
                stem = f"{date}_{t}"
                Image.new("RGB", (100, 50)).save(
                    raw_dir / "PKLot" / folder / f"{stem}.jpg"
                )
                (raw_dir / "PKLot" / folder / f"{stem}.xml").write_text(
                    _xml(
                        [
                            (' occupied="1"', 10, 10, 30, 30),  # kept, occupied
                            (' occupied="0"', 40, 10, 60, 30),  # kept, vacant
                            ("", 70, 10, 90, 30),  # no occupied attr -> dropped
                            (' occupied="1"', 95, 10, 200, 30),  # clipped to width
                        ]
                    )
                )
    # a frame with no XML must be skipped, not crash
    orphan = raw_dir / "PKLot" / "LOTA" / "Sunny" / "2011-01-01"
    Image.new("RGB", (100, 50)).save(orphan / "2012-09-01_23_00_00.jpg")


def test_prepare_coco_group_split_and_labels(tmp_path: Path) -> None:
    raw_dir, out = tmp_path / "PKLot", tmp_path / "canonical"
    _make_raw(raw_dir)

    PKLotAdapter().prepare_coco(raw_dir, out)

    date_to_splits: dict[tuple[str, str], set[str]] = {}
    lots_per_split: dict[str, set[str]] = {}
    total_images = 0
    for split in ("train", "valid", "test"):
        payload = json.loads((out / split / "_annotations.coco.json").read_text())
        assert [c["name"] for c in payload["categories"]] == ["vacant", "occupied"]
        assert {a["category_id"] for a in payload["annotations"]} <= {0, 1}
        # 4 kept per frame: 3 contour boxes + 1 rotatedRect-only fallback
        # (the space with no occupied attribute is dropped)
        assert len(payload["annotations"]) == 4 * len(payload["images"])
        for im in payload["images"]:
            lot, rest = im["file_name"].split("_", 1)
            date_to_splits.setdefault((lot, rest[:10]), set()).add(split)
            lots_per_split.setdefault(split, set()).add(lot)
            assert (out / split / im["file_name"]).exists()
        # the box past the right edge is clipped to the 100px width
        assert all(a["bbox"][0] + a["bbox"][2] <= 100.0 for a in payload["annotations"])
        total_images += len(payload["images"])

    assert total_images == 40  # orphan frame (no XML) excluded
    assert all(len(s) == 1 for s in date_to_splits.values())  # no day spans splits
    assert all(lots_per_split[s] == {"LOTA", "LOTB"} for s in lots_per_split)
