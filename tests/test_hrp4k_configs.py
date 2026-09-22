from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from detectionbench.utils.rfdetr import build_model_kwargs, build_training_kwargs

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def _compose(config_name: str, *overrides: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        return compose(config_name=config_name, overrides=list(overrides))


def test_hrp4k_rfdetr_config_builds_kwargs() -> None:
    cfg = _compose("hrp4k_rfdetr", "model.name=rfdetr-medium", "model.resolution=1024")
    training = build_training_kwargs(cfg)
    assert training["lr_scheduler"] == "cosine"
    assert training["warmup_epochs"] == 1.0
    assert training["resolution"] == 1024
    assert training["progress_bar"] == "tqdm"
    assert training["output_dir"].endswith("experiments/hrp4k/rfdetr-medium")
    assert build_model_kwargs(cfg, device_key="training")["num_classes"] == 1


def test_hrp4k_rfdetr_resolution_is_multiple_of_64() -> None:
    assert _compose("hrp4k_rfdetr").model.resolution % 64 == 0


def test_hrp4k_yolo_config_has_extra_args_and_high_imgsz() -> None:
    cfg = _compose("hrp4k_yolo")
    assert cfg.training.imgsz == 1536
    assert cfg.training.extra_args.close_mosaic == 15
    assert cfg.training.resume is None
    assert cfg.dataset.name == "hrp4k"
    assert cfg.model.num_classes == 1
