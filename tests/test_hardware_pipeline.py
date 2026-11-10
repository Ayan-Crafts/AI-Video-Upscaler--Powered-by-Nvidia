from pathlib import Path

import pytest

from app.core.config import ProcessingConfig
from app.core.pipeline import PipelineNotReadyError, validate_ai_pipeline
from app.hardware.gpu import GPUInfo
from app.hardware.system import system_metrics


def test_gpu_info_default() -> None:
    info = GPUInfo()
    assert not info.available
    assert info.backend == "unavailable"


def test_system_metrics_contract() -> None:
    metrics = system_metrics()
    assert {"cpu_percent", "ram_used_gb", "ram_total_gb", "ram_percent"} <= metrics.keys()


def test_ai_pipeline_fails_closed_before_ai_engines(tmp_path: Path) -> None:
    cfg = ProcessingConfig(tmp_path / "in.mp4", tmp_path / "out.mp4")
    with pytest.raises(PipelineNotReadyError, match="RIFE CUDA"):
        validate_ai_pipeline(cfg)
