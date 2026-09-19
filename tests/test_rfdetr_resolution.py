import json

from detectionbench.utils.rfdetr import read_training_resolution


def test_reads_resolution_next_to_checkpoint(tmp_path):
    (tmp_path / "training_config.json").write_text(
        json.dumps({"model_config": {"resolution": 576}})
    )
    assert read_training_resolution(str(tmp_path / "checkpoint_best_total.pth")) == 576


def test_missing_config_returns_none(tmp_path):
    assert read_training_resolution(str(tmp_path / "checkpoint_best_total.pth")) is None


def test_malformed_config_returns_none(tmp_path):
    (tmp_path / "training_config.json").write_text("{not json")
    assert read_training_resolution(str(tmp_path / "checkpoint_best_total.pth")) is None


def test_no_checkpoint_returns_none():
    assert read_training_resolution(None) is None
