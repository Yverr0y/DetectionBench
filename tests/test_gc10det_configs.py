from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def _compose(config_name: str, *overrides: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        return compose(config_name=config_name, overrides=list(overrides))


def test_gc10det_yolo_config_has_modest_imgsz_and_generous_epochs() -> None:
    cfg = _compose("gc10det_yolo")
    assert (
        cfg.training.imgsz == 640
    )  # boxes are large; resolution isn't the bottleneck here
    assert (
        cfg.training.epochs == 180
    )  # small dataset (1,840 train images) needs more epochs
    assert cfg.training.patience == 25  # small val split (230 images) is noisier
    assert cfg.training.resume is None
    assert cfg.dataset.name == "gc10det"
    assert cfg.dataset.eval_split == "test"
    assert cfg.model.num_classes == 10


def test_gc10det_yolo_config_accepts_model_override() -> None:
    cfg = _compose("gc10det_yolo", "model.name=yolo26s")
    assert cfg.model.name == "yolo26s"


def test_gc10det_rfdetr_config_uses_family_default_resolutions() -> None:
    cfg = _compose("gc10det_rfdetr")
    assert (
        cfg.model.resolution == 512
    )  # rfdetr-small default; boxes are large, no bump needed
    assert cfg.training.epochs == 180
    assert cfg.training.early_stopping_patience == 25
    assert cfg.dataset.name == "gc10det"
    assert cfg.model.num_classes == 10


def test_gc10det_rfdetr_config_accepts_model_and_resolution_override() -> None:
    cfg = _compose("gc10det_rfdetr", "model.name=rfdetr-nano", "model.resolution=384")
    assert cfg.model.name == "rfdetr-nano"
    assert cfg.model.resolution == 384
