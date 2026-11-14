from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def release_cuda_cache(torch, device) -> None:
    """Return cached CUDA memory without letting cleanup mask the original error.

    After a driver-level out-of-memory error (raised by recent PyTorch as
    ``torch.AcceleratorError: CUDA error: out of memory``) the next CUDA call,
    including ``empty_cache()``, can fail with the same error. Cleanup must not
    replace the error the user needs to see, so failures here are only logged.
    """
    try:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    except RuntimeError as exc:
        logger.warning("Could not release cached CUDA memory: %s", exc)
