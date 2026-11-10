# Third-party notices

The application uses or adapts the following projects. Their licenses apply to the respective portions and are reproduced under `app/engines/vendor_licenses/`.

- Practical-RIFE / RIFE HDv3 inference code: MIT License, copyright hzwer. The v4.25 checkpoint is downloaded from the pinned `MonsterMMORPG/Practical-RIFE` Hugging Face revision `4bbe4c19fb5363333cca83b6e82bb65ee02d741b`; checkpoint publisher terms remain applicable.
- Real-ESRGAN: BSD 3-Clause License, copyright Xintao Wang and contributors.
- BasicSR RRDBNet architecture: Apache License 2.0, copyright the BasicSR contributors.
- PySide6 / Qt: LGPLv3 or commercial terms; deployment must comply with Qt licensing.
- PyTorch / torchvision: project licenses and separate NVIDIA CUDA runtime terms apply.
- FFmpeg: LGPL/GPL obligations depend on the exact binary build and enabled components. Redistribute only a build whose licensing obligations are met.
- nvidia-ml-py: NVIDIA Python bindings. The NVIDIA display/compute driver remains separately installed.
- NumPy: BSD-style license.
- psutil: BSD-3-Clause.

Model weights are downloaded from upstream at first use and are not committed to this repository. Their source URL, file size and computed SHA-256 are recorded in the user model cache. Where upstream does not publish an expected checksum, the computed digest is not an independent authenticity guarantee. Review each checkpoint's license and redistribution conditions before bundling it in an installer.
