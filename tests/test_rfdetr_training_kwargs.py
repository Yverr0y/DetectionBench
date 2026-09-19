from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from detectionbench.utils.rfdetr import build_training_kwargs

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def _compose(*overrides: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        return compose(config_name="rfdetr", overrides=["dataset=uavdt", *overrides])


def test_default_scheduler_is_cosine_without_warmup():
    kwargs = build_training_kwargs(_compose())
    assert kwargs["lr_scheduler"] == "cosine"
    assert kwargs["warmup_epochs"] == 0.0


def test_warmup_epochs_override_is_passed_through():
    kwargs = build_training_kwargs(_compose("training.warmup_epochs=1.5"))
    assert kwargs["warmup_epochs"] == 1.5
