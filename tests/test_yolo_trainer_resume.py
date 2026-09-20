from pathlib import Path
from typing import Any

import pytest

from detectionbench.utils.trainer import YOLOTrainer


class _FakeYOLO:
    """Stands in for ultralytics.YOLO: records how it was built and trained."""

    instances: list["_FakeYOLO"] = []
    checkpoint_state: dict[str, Any] = {}

    def __init__(self, source: str) -> None:
        self.source = source
        self.ckpt = dict(self.checkpoint_state)
        self.train_kwargs: dict[str, Any] = {}
        _FakeYOLO.instances.append(self)

    def train(self, **kwargs: Any) -> str:
        self.train_kwargs = kwargs
        return "results"


@pytest.fixture
def trainer() -> YOLOTrainer:
    _FakeYOLO.instances = []
    _FakeYOLO.checkpoint_state = {"epoch": 7, "optimizer": {"state": 1}}
    instance = YOLOTrainer("yolov8n", num_classes=3, device="cpu")
    instance._UltralyticsYOLO = _FakeYOLO  # type: ignore[assignment]
    return instance


def _make_last(output_dir: Path) -> Path:
    weights = output_dir / "yolov8n" / "weights"
    weights.mkdir(parents=True)
    last = weights / "last.pt"
    last.write_bytes(b"x")
    return last


def test_fresh_run_is_unchanged(trainer: YOLOTrainer, tmp_path: Path) -> None:
    trainer.train("data.yaml", output_dir=tmp_path)
    fake = _FakeYOLO.instances[0]
    assert fake.source == "yolov8n.pt"
    assert "resume" not in fake.train_kwargs


@pytest.mark.parametrize("value", [True, "last", "true"])
def test_resume_true_uses_last_pt_in_output_dir(
    trainer: YOLOTrainer, tmp_path: Path, value: object
) -> None:
    last = _make_last(tmp_path)
    result = trainer.train("data.yaml", output_dir=tmp_path, resume=value)  # type: ignore[arg-type]
    fake = _FakeYOLO.instances[0]
    assert fake.source == str(last)
    assert fake.train_kwargs["resume"] is True
    assert result["output_dir"] == str(last.parent.parent)


def test_resume_accepts_explicit_checkpoint_path(
    trainer: YOLOTrainer, tmp_path: Path
) -> None:
    ckpt = tmp_path / "elsewhere" / "last.pt"
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"x")
    trainer.train("data.yaml", output_dir=tmp_path / "out", resume=str(ckpt))
    assert _FakeYOLO.instances[0].source == str(ckpt.resolve())


def test_missing_checkpoint_raises_clear_error(
    trainer: YOLOTrainer, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError, match="Cannot resume"):
        trainer.train("data.yaml", output_dir=tmp_path, resume=True)


def test_finished_run_checkpoint_is_refused(
    trainer: YOLOTrainer, tmp_path: Path
) -> None:
    _make_last(tmp_path)
    _FakeYOLO.checkpoint_state = {"epoch": -1, "optimizer": None}  # stripped at the end
    with pytest.raises(ValueError, match="not a resumable checkpoint"):
        trainer.train("data.yaml", output_dir=tmp_path, resume=True)
    assert not _FakeYOLO.instances[0].train_kwargs  # never trained
