from __future__ import annotations

import importlib

import numpy as np
import pytest

from app.core.frame_schedule import as_rate, intermediate_timesteps

pytest.importorskip("torch")
RRDBNet = importlib.import_module("app.engines.realesrgan_arch").RRDBNet
IFNet = importlib.import_module("app.engines.rife_vendor.ifnet_hdv3").IFNet


def test_frame_rate_fraction_parsing() -> None:
    assert as_rate("30000/1001") == pytest.approx(29.97002997, rel=1e-8)
    assert as_rate(29.44) == as_rate("736/25")


def test_rife_target_timestamps_are_inside_source_interval() -> None:
    samples = list(intermediate_timesteps("736/25", 60, 0))
    assert len(samples) == 2
    timestamps, alphas = zip(*samples)
    assert timestamps[0] < timestamps[1]
    assert all(0.0 < alpha < 1.0 for alpha in alphas)
    assert all(timestamp < as_rate("736/25") ** -1 for timestamp in timestamps)


def test_rife_schedule_has_no_duplicate_endpoint() -> None:
    samples = list(intermediate_timesteps(30, 60, 0))
    assert [round(alpha, 6) for _, alpha in samples] == [0.5]


def test_rife_ifnet_inference_shape_cpu() -> None:
    import torch

    network = IFNet().eval()
    frame0 = torch.rand(1, 3, 128, 128)
    frame1 = torch.rand(1, 3, 128, 128)
    with torch.inference_mode():
        output = network.interpolate(frame0, frame1, 0.5, (16, 8, 4, 2, 1))
    assert output.shape == frame0.shape
    assert output.dtype == frame0.dtype
    assert torch.isfinite(output).all()


def test_realesrgan_rrdbnet_x4_shape_cpu() -> None:
    import torch

    network = RRDBNet(num_feat=16, num_block=1, num_grow_ch=8).eval()
    frame = torch.rand(1, 3, 16, 20)
    with torch.inference_mode():
        output = network(frame)
    assert output.shape == (1, 3, 64, 80)
    assert torch.isfinite(output).all()


def test_real_esrgan_engine_rejects_invalid_rgb_shape_without_cuda() -> None:
    from app.engines.realesrgan import RealESRGANEngine

    with pytest.raises(ValueError, match="uint8 NumPy RGB"):
        RealESRGANEngine._validate_rgb(np.zeros((16, 16, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="H×W×3"):
        RealESRGANEngine._validate_rgb(np.zeros((16, 16), dtype=np.uint8))


def test_realesrgan_rrdbnet_x2_shape_cpu() -> None:
    import torch

    network = RRDBNet(num_feat=16, num_block=1, num_grow_ch=8, scale=2).eval()
    frame = torch.rand(1, 3, 16, 20)
    with torch.inference_mode():
        output = network(frame)
    assert output.shape == (1, 3, 32, 40)
    assert torch.isfinite(output).all()


def test_pipeline_can_skip_engines_when_source_already_matches_target() -> None:
    from app.core.ai_pipeline import AIFramePipeline

    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    with AIFramePipeline(interpolate=False, upscale=False, output_size=(32, 24)) as pipeline:
        prepared = pipeline.prepare_source_frame(frame)
        selected = pipeline.process_pair(frame, frame, 0.25)
    assert np.array_equal(prepared, frame)
    assert np.array_equal(selected, frame)


def test_realesrgan_x2_tiled_inference_handles_odd_dimensions_on_cpu() -> None:
    import torch
    from torch import nn

    from app.engines.realesrgan import RealESRGANEngine

    engine = RealESRGANEngine.__new__(RealESRGANEngine)
    engine.torch = torch
    engine.device = torch.device("cpu")
    engine.model_scale = 2
    engine.use_fp16 = False
    engine.tile_pad = 1
    engine.tile_size = 4
    engine.model = nn.Upsample(scale_factor=2, mode="nearest")

    tensor = torch.full((1, 3, 7, 9), 0.5, dtype=torch.float32)
    output_tensor = engine._upscale_tiles_tensor(tensor, tile_size=4)
    output = output_tensor[0].permute(1, 2, 0).contiguous().numpy()

    assert output.shape == (14, 18, 3)
    assert output.dtype == np.uint8
    assert np.all(output == 128)


def test_realesrgan_fit_to_canvas_preserves_aspect_ratio() -> None:
    import torch

    from app.engines.realesrgan import RealESRGANEngine

    engine = RealESRGANEngine.__new__(RealESRGANEngine)
    engine.torch = torch
    engine.device = torch.device("cpu")
    source = np.full((4, 8, 3), 200, dtype=np.uint8)

    output = engine._fit_to_canvas(source, target_width=8, target_height=8)

    assert output.shape == (8, 8, 3)
    assert np.all(output[:2] == 0)
    assert np.all(output[2:6] == 200)
    assert np.all(output[6:] == 0)


def test_cuda_frame_pipeline_does_not_import_models_when_skipped() -> None:
    from app.core.ai_pipeline import AIFramePipeline

    with AIFramePipeline(
        interpolate=False, upscale=False, output_size=(32, 24)
    ) as pipeline:
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        result = pipeline.process_pair(frame, frame, 0.25)

    assert np.array_equal(result, frame)


def test_ai_pipeline_tensor_canvas_preserves_aspect_ratio() -> None:
    import torch

    from app.core.ai_pipeline import AIFramePipeline

    tensor = torch.ones((1, 3, 4, 8), dtype=torch.float32)
    output = AIFramePipeline._tensor_to_canvas(tensor, (8, 8))

    assert output.shape == (8, 8, 3)
    assert np.all(output[:2] == 0)
    assert np.all(output[2:6] == 255)
    assert np.all(output[6:] == 0)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA hardware required")
def test_rife_to_realesrgan_pipeline_keeps_intermediate_on_cuda() -> None:
    import torch

    from app.core.ai_pipeline import AIFramePipeline

    class FakeRife:
        def interpolate_tensor(self, frame0, frame1, timestep):
            assert frame0.is_cuda and frame1.is_cuda
            return frame0 * (1.0 - timestep) + frame1 * timestep

    class FakeUpscaler:
        def upscale_tensor(self, tensor, output_size):
            assert tensor.is_cuda
            assert output_size == (32, 24)
            return np.zeros((24, 32, 3), dtype=np.uint8)

    pipeline = AIFramePipeline.__new__(AIFramePipeline)
    pipeline.output_size = (32, 24)
    pipeline.device_index = 0
    pipeline.rife = FakeRife()
    pipeline.upscaler = FakeUpscaler()
    frame0 = np.zeros((24, 32, 3), dtype=np.uint8)
    frame1 = np.full((24, 32, 3), 255, dtype=np.uint8)

    output = pipeline.process_pair(frame0, frame1, 0.5)

    assert output.shape == (24, 32, 3)
    assert torch.cuda.is_available()


def test_realesrgan_upscale_returns_copy_when_target_matches_input() -> None:
    from app.engines.realesrgan import RealESRGANEngine

    engine = RealESRGANEngine.__new__(RealESRGANEngine)
    frame = np.full((12, 20, 3), 127, dtype=np.uint8)
    output = engine.upscale(frame, output_size=(20, 12))

    assert np.array_equal(output, frame)
    assert output is not frame
