from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from app.engines.realesrgan import RealESRGANEngine
from app.engines.rife import RIFEEngine


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and benchmark AI Upscaler Studio RIFE and Real-ESRGAN CUDA engines."
    )
    parser.add_argument("--device", type=int, default=0, help="PyTorch CUDA device index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Synthetic test-frame width")
    parser.add_argument("--height", type=int, default=360, help="Synthetic test-frame height")
    parser.add_argument("--tile", type=int, default=0, help="Real-ESRGAN tile size; 0 selects automatically")
    parser.add_argument("--fp32", action="store_true", help="Disable FP16 inference")
    parser.add_argument("--include-x4", action="store_true", help="Also download and test the x4plus checkpoint")
    parser.add_argument("--model-root", type=Path, default=None, help="Optional model-cache directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        import torch
    except ImportError as exc:
        parser.error(f"PyTorch is not installed: {exc}")

    if not torch.cuda.is_available():
        parser.error("PyTorch cannot access CUDA. Install a CUDA-enabled PyTorch wheel and check nvidia-smi.")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        parser.error(f"CUDA device index {args.device} is not available.")
    if args.width < 128 or args.height < 128:
        parser.error("Test frame dimensions must both be at least 128 pixels.")

    torch.cuda.set_device(args.device)
    device_name = torch.cuda.get_device_name(args.device)
    properties = torch.cuda.get_device_properties(args.device)
    capability = torch.cuda.get_device_capability(args.device)
    fp16 = not args.fp32 and capability[0] >= 7
    print(
        json.dumps(
            {
                "gpu": device_name,
                "compute_capability": f"{capability[0]}.{capability[1]}",
                "cuda_runtime": torch.version.cuda,
                "torch": torch.__version__,
                "vram_total_gb": round(properties.total_memory / 1024**3, 2),
                "fp16_enabled": fp16,
            },
            indent=2,
        )
    )

    rng = np.random.default_rng(42)
    frame0 = rng.integers(0, 256, size=(args.height, args.width, 3), dtype=np.uint8)
    frame1 = rng.integers(0, 256, size=(args.height, args.width, 3), dtype=np.uint8)
    results: dict[str, object] = {}

    rife = None
    sr_engines: list[RealESRGANEngine] = []
    try:
        print("\n[RIFE] Preparing model (first run downloads the checkpoint)...")
        rife = RIFEEngine(
            device_index=args.device,
            fp16=fp16,
            model_root=args.model_root,
            progress=lambda pct, stage: print(f"{pct:5.1f}% {stage}"),
        )
        frame0_tensor = torch.from_numpy(frame0).permute(2, 0, 1).unsqueeze(0)
        frame1_tensor = torch.from_numpy(frame1).permute(2, 0, 1).unsqueeze(0)
        frame0_tensor = frame0_tensor.to(device=f"cuda:{args.device}", dtype=torch.float32).div_(255.0)
        frame1_tensor = frame1_tensor.to(device=f"cuda:{args.device}", dtype=torch.float32).div_(255.0)

        torch.cuda.reset_peak_memory_stats(args.device)
        torch.cuda.synchronize(args.device)
        start = time.perf_counter()
        interpolated = rife.interpolate_tensor(frame0_tensor, frame1_tensor, 0.5)
        torch.cuda.synchronize(args.device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        results["rife"] = {
            "status": "passed",
            "output_shape": list(interpolated.shape),
            "latency_ms": round(elapsed_ms, 2),
            "effective_pair_fps": round(1000.0 / elapsed_ms, 3),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated(args.device) / 1024**3, 3),
            "effective_inference_scale": rife.inference_scale,
        }
        print(json.dumps(results["rife"], indent=2))

        model_scales = [2, 4] if args.include_x4 else [2]
        for model_scale in model_scales:
            print(f"\n[Real-ESRGAN x{model_scale}] Preparing model...")
            engine = RealESRGANEngine(
                device_index=args.device,
                fp16=fp16,
                tile_size=args.tile,
                model_scale=model_scale,
                model_root=args.model_root,
                progress=lambda pct, stage: print(f"{pct:5.1f}% {stage}"),
            )
            sr_engines.append(engine)
            torch.cuda.reset_peak_memory_stats(args.device)
            torch.cuda.synchronize(args.device)
            start = time.perf_counter()
            # Consume RIFE's CUDA tensor directly; no intermediate CUDA->CPU->CUDA copy.
            upscaled = engine.upscale_tensor(interpolated)
            torch.cuda.synchronize(args.device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            results[f"realesrgan_x{model_scale}"] = {
                "status": "passed",
                "output_shape": list(upscaled.shape),
                "latency_ms": round(elapsed_ms, 2),
                "effective_frames_per_second": round(1000.0 / elapsed_ms, 3),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated(args.device) / 1024**3, 3),
                "tile_size_after_oom_adaptation": engine.tile_size,
                "precision": "fp16" if engine.use_fp16 else "fp32",
                "input_handoff": "CUDA tensor (no CPU round-trip from RIFE)",
            }
            print(json.dumps(results[f"realesrgan_x{model_scale}"], indent=2))
            del upscaled
            engine.close()
            sr_engines.remove(engine)

        print("\nBENCHMARK_RESULT=" + json.dumps(results, indent=2))
        return 0
    finally:
        for engine in sr_engines:
            engine.close()
        if rife is not None:
            rife.close()
        if "frame0_tensor" in locals():
            del frame0_tensor
        if "frame1_tensor" in locals():
            del frame1_tensor
        if "interpolated" in locals():
            del interpolated
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
