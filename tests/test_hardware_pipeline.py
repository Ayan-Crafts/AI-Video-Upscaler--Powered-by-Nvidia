from pathlib import Path

import pytest

from app.core.config import ProcessingConfig
from app.core.pipeline import PipelineNotReadyError, validate_ai_pipeline
from app.engines.availability import EngineAvailability
from app.hardware.gpu import GPUInfo
from app.hardware.system import system_metrics


def test_gpu_info_default() -> None:
    info = GPUInfo()
    assert not info.available
    assert info.backend == "unavailable"


def test_system_metrics_contract() -> None:
    metrics = system_metrics()
    assert {"cpu_percent", "ram_used_gb", "ram_total_gb", "ram_percent"} <= metrics.keys()


def test_ai_pipeline_fails_closed_when_rife_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The engine availability check now correctly reports CUDA readiness on a
    # supported GPU. Simulate RIFE being unavailable to verify fail-closed logic.
    monkeypatch.setattr(
        "app.core.pipeline.rife_status",
        lambda _device_index=0: EngineAvailability(
            "RIFE CUDA", False, "RIFE checkpoint unavailable"
        ),
    )
    monkeypatch.setattr(
        "app.core.pipeline.realesrgan_status",
        lambda _device_index=0: EngineAvailability(
            "Real-ESRGAN CUDA", True, "CUDA ready"
        ),
    )

    config = ProcessingConfig(tmp_path / "input.mp4", tmp_path / "output.mp4")
    with pytest.raises(PipelineNotReadyError, match="RIFE CUDA"):
        validate_ai_pipeline(config)
