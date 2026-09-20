from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from detectionbench.utils.rfdetr import build_model_kwargs, build_training_kwargs

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def _compose(config_name: str, *overrides: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        return compose(config_name=config_name, overrides=list(overrides))


def test_visdrone_rfdetr_config_builds_kwargs():
    cfg = _compose(
        "visdrone_rfdetr", "model.name=rfdetr-medium", "model.resolution=896"
    )
    training = build_training_kwargs(cfg)
    assert training["lr_scheduler"] == "cosine"
    assert training["warmup_epochs"] == 1.0
    assert training["resolution"] == 896
    assert training["output_dir"].endswith("experiments/visdrone/rfdetr-medium")
    assert build_model_kwargs(cfg, device_key="training")["num_classes"] == 11


def test_visdrone_rfdetr_resolution_is_multiple_of_64():
    assert _compose("visdrone_rfdetr").model.resolution % 64 == 0


def test_visdrone_yolo_config_has_extra_args_and_high_imgsz():
    cfg = _compose("visdrone_yolo")
    assert cfg.training.imgsz == 1280
    assert cfg.training.extra_args.max_det == 500
    assert cfg.dataset.name == "visdrone"


def test_default_yolo_config_has_no_extra_args():
    assert _compose("config").training.get("extra_args") is None
