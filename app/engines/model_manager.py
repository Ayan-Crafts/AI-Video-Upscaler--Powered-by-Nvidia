from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from app.paths import legacy_model_dirs, model_dir
from app.version import USER_AGENT

logger = logging.getLogger(__name__)


ProgressCallback = Callable[[float, str], None]

# A partial download whose file has not been written for this long belongs to
# a crashed or killed process and is safe to delete.
STALE_PART_SECONDS = 15 * 60


def default_model_root() -> Path:
    """Per-user model cache (see :mod:`app.paths`)."""
    return model_dir()


class ModelDownloadError(RuntimeError):
    pass


class ModelManager:
    """Download model files atomically and verify their integrity.

    * Downloads stream into a ``.part`` file in the cache directory and are
      atomically renamed only after size and SHA-256 checks pass.
    * Interrupted downloads leave no partial checkpoint; stale ``.part`` files
      from crashed processes are removed on the next attempt.
    * Network failures are retried with a short back-off.
    * Checkpoints from the pre-1.0 developer cache are reused instead of being
      downloaded again.

    Model files are never unpickled from arbitrary URLs. The inference loaders
    use torch.load(weights_only=True) and strict state-dict validation.
    """

    def __init__(
        self,
        root: Path | None = None,
        *,
        legacy_roots: list[Path] | None = None,
        retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.root = root or default_model_root()
        if legacy_roots is None:
            legacy_roots = legacy_model_dirs() if root is None else []
        self.legacy_roots = [path for path in legacy_roots if path != self.root]
        self.retries = max(1, retries)
        self.retry_delay = max(0.0, retry_delay)

    def ensure_file(
        self,
        *,
        relative_path: str,
        url: str,
        min_bytes: int,
        expected_sha256: str | None = None,
        expected_bytes: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> Path:
        destination = self.root / relative_path
        metadata_path = destination.with_suffix(destination.suffix + ".json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._remove_stale_parts(destination)

        if not destination.is_file():
            self._import_legacy_checkpoint(relative_path, destination, metadata_path, min_bytes)

        mismatch_message: str | None = None
        if destination.is_file():
            cached = self._validate_cached(
                destination, metadata_path, min_bytes, expected_sha256, expected_bytes
            )
            if cached is True:
                return destination
            if isinstance(cached, str):
                mismatch_message = cached

        try:
            return self._download(
                destination, metadata_path, url, min_bytes, expected_sha256, expected_bytes, progress
            )
        except ModelDownloadError as exc:
            if mismatch_message:
                raise ModelDownloadError(
                    f"{mismatch_message} The file was removed and could not be downloaded again: {exc}"
                ) from exc
            raise

    # ------------------------------------------------------------------ cache
    def _validate_cached(
        self,
        destination: Path,
        metadata_path: Path,
        min_bytes: int,
        expected_sha256: str | None,
        expected_bytes: int | None,
    ) -> bool | str:
        """Return True if the cached file is usable, else a reason (file removed)."""
        size = destination.stat().st_size
        if size < min_bytes or (expected_bytes is not None and size != expected_bytes):
            logger.warning("Cached model %s has unexpected size %s; re-downloading.", destination, size)
            self._discard(destination, metadata_path)
            return False

        actual_sha = self.sha256(destination)
        if expected_sha256 and actual_sha.lower() != expected_sha256.lower():
            self._discard(destination, metadata_path)
            return f"Cached model checksum mismatch for {destination.name}."

        cached_metadata: dict[str, object] = {}
        if metadata_path.is_file():
            try:
                cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cached_metadata = {}
        recorded_sha = cached_metadata.get("sha256")
        if recorded_sha and actual_sha.lower() != str(recorded_sha).lower():
            logger.warning(
                "Cached model integrity check failed for %s; it will be downloaded again.",
                destination,
            )
            self._discard(destination, metadata_path)
            return f"Cached model integrity check failed for {destination.name}."
        if not recorded_sha:
            # A manually copied checkpoint has no prior integrity record. Record
            # the observed digest, but do not claim upstream verification.
            self._write_metadata(
                metadata_path,
                {
                    "file": destination.name,
                    "source": "pre-existing local checkpoint",
                    "sha256": actual_sha,
                    "size_bytes": size,
                    "checksum_status": (
                        "verified" if expected_sha256 else "computed-local-no-published-checksum"
                    ),
                },
            )
        return True

    def _import_legacy_checkpoint(
        self, relative_path: str, destination: Path, metadata_path: Path, min_bytes: int
    ) -> None:
        for legacy_root in self.legacy_roots:
            legacy_file = legacy_root / relative_path
            try:
                if not legacy_file.is_file() or legacy_file.stat().st_size < min_bytes:
                    continue
                fd, temp_name = tempfile.mkstemp(
                    prefix=destination.name + ".", suffix=".part", dir=destination.parent
                )
                os.close(fd)
                try:
                    shutil.copyfile(legacy_file, temp_name)
                    os.replace(temp_name, destination)
                finally:
                    Path(temp_name).unlink(missing_ok=True)
                legacy_metadata = legacy_file.with_suffix(legacy_file.suffix + ".json")
                if legacy_metadata.is_file():
                    shutil.copyfile(legacy_metadata, metadata_path)
                logger.info("Reused cached checkpoint from previous build: %s", legacy_file)
                return
            except OSError as exc:
                logger.warning("Could not reuse legacy checkpoint %s: %s", legacy_file, exc)

    @staticmethod
    def _remove_stale_parts(destination: Path) -> None:
        now = time.time()
        for part in destination.parent.glob(destination.name + ".*.part"):
            try:
                if now - part.stat().st_mtime > STALE_PART_SECONDS:
                    part.unlink()
                    logger.info("Removed interrupted download %s", part)
            except OSError:
                continue

    @staticmethod
    def _discard(destination: Path, metadata_path: Path) -> None:
        destination.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

    @staticmethod
    def _write_metadata(metadata_path: Path, payload: dict[str, object]) -> None:
        try:
            metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not write model metadata %s: %s", metadata_path, exc)

    # --------------------------------------------------------------- download
    def _download(
        self,
        destination: Path,
        metadata_path: Path,
        url: str,
        min_bytes: int,
        expected_sha256: str | None,
        expected_bytes: int | None,
        progress: ProgressCallback | None,
    ) -> Path:
        last_error: ModelDownloadError | None = None
        for attempt in range(1, self.retries + 1):
            try:
                return self._download_once(
                    destination, metadata_path, url, min_bytes, expected_sha256, expected_bytes, progress
                )
            except ModelDownloadError as exc:
                last_error = exc
                if attempt < self.retries:
                    delay = self.retry_delay * attempt
                    logger.warning(
                        "Download attempt %s/%s for %s failed (%s); retrying in %.0f s.",
                        attempt, self.retries, destination.name, exc, delay,
                    )
                    if progress:
                        progress(0.0, f"Retrying download of {destination.name}")
                    time.sleep(delay)
        assert last_error is not None
        raise last_error

    def _download_once(
        self,
        destination: Path,
        metadata_path: Path,
        url: str,
        min_bytes: int,
        expected_sha256: str | None,
        expected_bytes: int | None,
        progress: ProgressCallback | None,
    ) -> Path:
        destination.unlink(missing_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
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

            if total > 0 and downloaded != total:
                raise ModelDownloadError(
                    f"Download of {destination.name} was interrupted ({downloaded:,} of {total:,} bytes)."
                )
            if downloaded < min_bytes:
                raise ModelDownloadError(
                    f"Downloaded model is unexpectedly small ({downloaded:,} bytes): {destination.name}"
                )
            if expected_bytes is not None and downloaded != expected_bytes:
                raise ModelDownloadError(
                    f"Downloaded model {destination.name} has {downloaded:,} bytes; "
                    f"expected exactly {expected_bytes:,}."
                )
            actual_sha = digest.hexdigest()
            if expected_sha256 and actual_sha.lower() != expected_sha256.lower():
                raise ModelDownloadError(
                    f"SHA-256 mismatch for {destination.name}: expected {expected_sha256}, got {actual_sha}"
                )
            os.replace(temp_path, destination)
            self._write_metadata(
                metadata_path,
                {
                    "file": destination.name,
                    "source": url,
                    "sha256": actual_sha,
                    "size_bytes": downloaded,
                    "checksum_status": "verified" if expected_sha256 else "computed-no-published-checksum",
                },
            )
            if progress:
                progress(100.0, f"Model ready: {destination.name}")
            logger.info("Model ready: %s (sha256=%s)", destination, actual_sha)
            return destination
        except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
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
