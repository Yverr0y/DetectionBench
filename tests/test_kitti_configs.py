from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def _compose(config_name: str, *overrides: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        return compose(config_name=config_name, overrides=list(overrides))


def test_kitti_yolo_config_has_high_imgsz_and_generous_patience() -> None:
    cfg = _compose("kitti_yolo")
    assert cfg.training.imgsz == 1280
    assert cfg.training.epochs == 100
    assert cfg.training.patience == 20
    assert cfg.training.resume is None
    assert cfg.dataset.name == "kitti"
    assert cfg.dataset.eval_split == "val"  # Ultralytics data.yaml key, not "valid"
    assert cfg.model.num_classes == 8


def test_kitti_yolo_config_accepts_model_override() -> None:
    cfg = _compose("kitti_yolo", "model.name=yolo26s")
    assert cfg.model.name == "yolo26s"
