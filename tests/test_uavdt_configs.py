from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from detectionbench.utils.rfdetr import build_model_kwargs, build_training_kwargs

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def _compose(config_name: str, *overrides: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        return compose(config_name=config_name, overrides=list(overrides))


def test_uavdt_rfdetr_config_builds_kwargs():
    cfg = _compose("uavdt_rfdetr", "model.name=rfdetr-medium", "model.resolution=768")
    training = build_training_kwargs(cfg)
    assert training["lr_scheduler"] == "cosine"
    assert training["warmup_epochs"] == 0.5
    assert training["resolution"] == 768
    assert "uavdt_v2" in training["output_dir"]
    assert build_model_kwargs(cfg, device_key="training")["num_classes"] == 3


def test_uavdt_rfdetr_resolution_is_multiple_of_64():
    assert _compose("uavdt_rfdetr").model.resolution % 64 == 0


def test_uavdt_yolo_config_has_extra_args():
    cfg = _compose("uavdt_yolo")
    assert cfg.training.imgsz == 1024
    assert cfg.training.extra_args.mixup == 0.1
    assert cfg.dataset.name == "uavdt"
    assert cfg.model.num_classes == 3
