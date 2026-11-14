"""Fail the build if the environment would package a CPU-only PyTorch.

Run with the build environment's Python before PyInstaller. Optionally pass a
built bundle directory to confirm the CUDA runtime libraries were collected.

Usage: python verify_cuda_runtime.py [--expected-cuda 12.6] [--bundle DIR] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Library name stems that must be present in a bundle for CUDA inference.
WINDOWS_LIBS = ("cudart64_", "cublas64_", "cublasLt64_", "cudnn64_", "cufft64_", "nvrtc64_",
                "torch_cuda.dll", "c10_cuda.dll")
LINUX_LIBS = ("libcudart.so", "libcublas.so", "libcublasLt.so", "libcudnn.so", "libcufft.so",
              "libnvrtc.so", "libtorch_cuda.so", "libc10_cuda.so", "libnccl.so")


def environment_report(expected_cuda: str) -> dict[str, object]:
    import torch

    report: dict[str, object] = {
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.version.cuda else None,
        "arch_list": torch.cuda.get_arch_list() if torch.version.cuda else [],
        "cuda_available_on_build_machine": torch.cuda.is_available(),
    }
    report["ok"] = report["cuda_runtime"] == expected_cuda and "+cpu" not in torch.__version__
    return report


def bundle_report(bundle: Path) -> dict[str, object]:
    names = [p.name for p in bundle.rglob("*") if p.is_file()]
    stems = WINDOWS_LIBS if sys.platform.startswith("win") else LINUX_LIBS
    missing = [stem for stem in stems if not any(name.startswith(stem) for name in names)]
    return {"missing_cuda_libraries": missing, "ok": not missing}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-cuda", default="12.6")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report: dict[str, object] = {}
    if args.bundle:
        report["bundle"] = bundle_report(args.bundle)
    else:
        report["environment"] = environment_report(args.expected_cuda)
    text = json.dumps(report, indent=2)
    if args.json:
        args.json.write_text(text, encoding="utf-8")
    print(text)
    if not all(section["ok"] for section in report.values()):
        sys.exit("CUDA runtime verification FAILED")


if __name__ == "__main__":
    main()
