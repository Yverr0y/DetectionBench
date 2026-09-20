import json
import sys
from pathlib import Path

import pytest

from detectionbench.scripts.evaluate import parse_args


def _parse(monkeypatch: pytest.MonkeyPatch, *argv: str):
    monkeypatch.setattr(sys, "argv", ["detectionbench-evaluate", *argv])
    return parse_args()


def test_rfdetr_config_supplies_paths_dataset_and_split(monkeypatch, tmp_path) -> None:
    args = _parse(
        monkeypatch, "--config-name", "uavdt_rfdetr", "--model", "rfdetr-nano"
    )
    assert args.dataset == "uavdt"
    assert args.split == "test"
    assert args.checkpoint == "experiments/uavdt/rfdetr-nano/checkpoint_best_total.pth"
    assert args.output_dir == "experiments/uavdt/rfdetr-nano/evaluation"


def test_config_accepts_a_path_with_extension(monkeypatch) -> None:
    args = _parse(
        monkeypatch,
        "--config-name",
        "configs/visdrone_rfdetr.yaml",
        "--model",
        "rfdetr-small",
    )
    assert args.dataset == "visdrone"
    assert args.output_dir == "experiments/visdrone/rfdetr-small/evaluation"


def test_yolo_config_uses_its_evaluation_section(monkeypatch) -> None:
    args = _parse(monkeypatch, "--config-name", "bdd100k_yolo", "--model", "yolov8n")
    assert args.dataset == "bdd100k"
    assert args.checkpoint.endswith("bdd100k/yolov8n/yolov8n/weights/best.pt")
    assert args.output_dir.endswith("bdd100k/yolov8n/yolov8n/evaluation")


def test_explicit_flags_win_over_config(monkeypatch) -> None:
    args = _parse(
        monkeypatch,
        "--config-name",
        "uavdt_rfdetr",
        "--model",
        "rfdetr-nano",
        "--checkpoint",
        "my.pth",
        "--output-dir",
        "out",
        "--split",
        "valid",
    )
    assert (args.checkpoint, args.output_dir, args.split) == ("my.pth", "out", "valid")


def test_family_mismatch_is_an_error(monkeypatch) -> None:
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--config-name", "uavdt_yolo", "--model", "rfdetr-nano")


def test_training_config_json_beats_the_config_resolution(
    monkeypatch, tmp_path: Path
) -> None:
    ckpt = tmp_path / "checkpoint_best_total.pth"
    (tmp_path / "training_config.json").write_text(
        json.dumps({"model_config": {"resolution": 640}})
    )
    args = _parse(
        monkeypatch,
        "--config-name",
        "uavdt_rfdetr",
        "--model",
        "rfdetr-nano",
        "--checkpoint",
        str(ckpt),
    )
    # Left unset so evaluate_rfdetr reads the 640 the model was actually trained at,
    # not the config's 704 default.
    assert args.resolution is None


def test_config_resolution_is_the_fallback_without_training_config(
    monkeypatch, tmp_path: Path
) -> None:
    args = _parse(
        monkeypatch,
        "--config-name",
        "uavdt_rfdetr",
        "--model",
        "rfdetr-nano",
        "--checkpoint",
        str(tmp_path / "checkpoint_best_total.pth"),
    )
    assert args.resolution == 704


def test_without_config_the_required_flags_still_apply(monkeypatch) -> None:
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--model", "yolov8n", "--dataset", "lisa")  # no checkpoint
    args = _parse(
        monkeypatch,
        "--model",
        "yolov8n",
        "--dataset",
        "lisa",
        "--checkpoint",
        "w.pt",
    )
    assert args.split == "test"
