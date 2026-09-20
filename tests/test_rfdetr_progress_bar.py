from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from detectionbench.utils.rfdetr import build_training_kwargs

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize(
    "config_name", ["rfdetr", "visdrone_rfdetr", "bdd100k_rfdetr", "uavdt_rfdetr"]
)
def test_rfdetr_configs_use_tqdm_progress_bar(config_name: str) -> None:
    # "rich" draws nothing when stdout is piped (e.g. `| tee`); tqdm does.
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        cfg = compose(config_name=config_name)
    assert build_training_kwargs(cfg)["progress_bar"] == "tqdm"
