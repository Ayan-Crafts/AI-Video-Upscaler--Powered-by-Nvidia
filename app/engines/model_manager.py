from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


ProgressCallback = Callable[[float, str], None]


def default_model_root() -> Path:
    """Return the per-user model cache directory on Windows and other OSes."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "NVIDIA AI Video Studio" / "models"


class ModelDownloadError(RuntimeError):
    pass


class ModelManager:
    """Download model files atomically and record their observed SHA-256.

    Model files are never unpickled from arbitrary URLs. The inference loaders
    use torch.load(weights_only=True) and strict state-dict validation.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or default_model_root()

    def ensure_file(
        self,
        *,
        relative_path: str,
        url: str,
        min_bytes: int,
        expected_sha256: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> Path:
        destination = self.root / relative_path
        metadata_path = destination.with_suffix(destination.suffix + ".json")
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.is_file() and destination.stat().st_size >= min_bytes:
            actual_sha = self.sha256(destination)
            if expected_sha256 and actual_sha.lower() != expected_sha256.lower():
                destination.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)
                raise ModelDownloadError(
                    f"Cached model checksum mismatch for {destination.name}. The file was removed."
                )
            if metadata_path.is_file():
                try:
                    cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    cached_metadata = {}
                recorded_sha = cached_metadata.get("sha256")
                if recorded_sha and actual_sha.lower() != str(recorded_sha).lower():
                    destination.unlink(missing_ok=True)
                    metadata_path.unlink(missing_ok=True)
                    logger.warning(
                        "Cached model integrity check failed for %s; it will be downloaded again.",
                        destination,
                    )
                elif not recorded_sha:
                    metadata_path.write_text(
                        json.dumps(
                            {
                                "file": destination.name,
                                "source": "pre-existing local checkpoint",
                                "sha256": actual_sha,
                                "size_bytes": destination.stat().st_size,
                                "checksum_status": "computed-local-no-published-checksum",
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    return destination
                else:
                    return destination
            else:
                # A manually copied checkpoint has no prior integrity record.
                # Record the observed digest, but do not claim upstream verification.
                metadata_path.write_text(
                    json.dumps({
                        "file": destination.name,
                        "source": "pre-existing local checkpoint",
                        "sha256": actual_sha,
                        "size_bytes": destination.stat().st_size,
                        "checksum_status": "computed-local-no-published-checksum",
                    }, indent=2),
                    encoding="utf-8",
                )
                return destination

        destination.unlink(missing_ok=True)
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "NVIDIA-AI-Video-Studio/0.2 model-manager"},
        )
        fd, temp_name = tempfile.mkstemp(
            prefix=destination.name + ".", suffix=".part", dir=destination.parent
        )
        os.close(fd)
        temp_path = Path(temp_name)
        digest = hashlib.sha256()
        downloaded = 0

        try:
            with urllib.request.urlopen(request, timeout=30) as response, temp_path.open("wb") as out:
                total = int(response.headers.get("Content-Length", "0") or 0)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    if progress and total > 0:
                        progress(min(100.0, downloaded * 100.0 / total), f"Downloading {destination.name}")
                out.flush()
                os.fsync(out.fileno())

            if downloaded < min_bytes:
                raise ModelDownloadError(
                    f"Downloaded model is unexpectedly small ({downloaded:,} bytes): {destination.name}"
                )
            actual_sha = digest.hexdigest()
            if expected_sha256 and actual_sha.lower() != expected_sha256.lower():
                raise ModelDownloadError(
                    f"SHA-256 mismatch for {destination.name}: expected {expected_sha256}, got {actual_sha}"
                )
            os.replace(temp_path, destination)
            metadata = {
                "file": destination.name,
                "source": url,
                "sha256": actual_sha,
                "size_bytes": downloaded,
                "checksum_status": "verified" if expected_sha256 else "computed-no-published-checksum",
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            if progress:
                progress(100.0, f"Model ready: {destination.name}")
            logger.info("Model ready: %s (sha256=%s)", destination, actual_sha)
            return destination
        except (OSError, urllib.error.URLError, TimeoutError, ModelDownloadError) as exc:
            temp_path.unlink(missing_ok=True)
            if isinstance(exc, ModelDownloadError):
                raise
            raise ModelDownloadError(f"Could not download {destination.name}: {exc}") from exc
        finally:
            temp_path.unlink(missing_ok=True)

    @staticmethod
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
