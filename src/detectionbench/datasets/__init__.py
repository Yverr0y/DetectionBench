"""Dataset registry: importing this package registers every known adapter."""

from detectionbench.datasets import (  # noqa: F401
    brackish,
    doclaynet,
    duo,
    exdark,
    gc10det,
    gwhd,
    hrsid,
    lisa,
    llvip,
    marida,
    neudet,
    publaynet,
    rdd2022,
    seadronessee,
    seaships,
    sku110k,
    ssdd,
    uavdt,
    visdrone,
)
from detectionbench.datasets.registry import get, get_spec, list_datasets

__all__ = ["get", "get_spec", "list_datasets"]
