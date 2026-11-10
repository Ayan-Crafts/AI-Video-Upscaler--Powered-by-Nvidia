from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

from app.core.ai_pipeline import AIFramePipeline
from app.core.config import ProcessingConfig, validate_config
from app.core.frame_schedule import as_rate
from app.media.ffmpeg import discover_ffmpeg, probe_media


class AIVideoExecutionError(RuntimeError):
    pass


ProgressCallback = Callable[[float, str], None]


def _read_exact(stream, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if not chunks:
                return None
            raise AIVideoExecutionError("FFmpeg returned an incomplete raw video frame.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _write_all(stream, data: bytes) -> None:
    """Write every byte to an unbuffered subprocess pipe or raise on closure."""
    view = memoryview(data)
    while view:
        written = stream.write(view)
        if written is None or written <= 0:
            raise BrokenPipeError("The FFmpeg encoder pipe closed during a frame write.")
        view = view[written:]


def _rate_from_stream(value: str) -> Fraction:
    try:
        return as_rate(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise AIVideoExecutionError(f"Invalid or missing input frame rate: {value!r}") from exc


def _format_rate(rate: Fraction) -> str:
    return f"{rate.numerator}/{rate.denominator}"


class AIVideoRunner:
    """CFR video runner: FFmpeg decode -> CUDA RIFE -> CUDA Real-ESRGAN -> NVENC.

    The first production pass intentionally rejects clearly variable-frame-rate
    input rather than silently desynchronizing audio. VFR timestamp support is a
    separate follow-up once frame-PTS transport is implemented end-to-end.
    """

    def __init__(self, ffmpeg_path: str | None = None) -> None:
        self.ffmpeg_path = ffmpeg_path or shutil.which("ffmpeg")
        self._cancel = threading.Event()
        self._decoder: subprocess.Popen[bytes] | None = None
        self._encoder: subprocess.Popen[bytes] | None = None

    def cancel(self) -> None:
        self._cancel.set()
        for process in (self._decoder, self._encoder):
            if process is not None and process.poll() is None:
                process.terminate()

    def _encoder_command(
        self,
        config: ProcessingConfig,
        source_width: int,
        source_height: int,
        target_rate: Fraction,
        partial_path: Path,
    ) -> list[str]:
        if not self.ffmpeg_path:
            raise AIVideoExecutionError("FFmpeg was not found. Install FFmpeg or bundle it with the app.")
        args = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", f"{config.output_width}x{config.output_height}",
            "-framerate", _format_rate(target_rate),
            "-i", "pipe:0",
            "-i", str(config.input_path),
            "-map", "0:v:0",
            "-map", "1:a?",
            "-map_metadata", "1",
            "-c:v", config.video_codec,
        ]
        if config.video_codec.endswith("_nvenc"):
            args.extend(["-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", "19", "-b:v", "0"])
        elif config.video_codec == "libx264":
            args.extend(["-preset", "medium", "-crf", "18"])
        elif config.video_codec == "libx265":
            args.extend(["-preset", "medium", "-crf", "20"])
        elif config.video_codec == "libvpx-vp9":
            args.extend(["-b:v", "0", "-crf", "32"])

        if config.container == "webm":
            args.extend(["-c:a", "libopus", "-b:a", "192k"])
        else:
            args.extend(["-c:a", "aac", "-b:a", "192k"])
        if config.resample_audio:
            args.extend(["-ar", str(config.audio_sample_rate)])
        output_pixel_format = "yuv422p10le" if config.video_codec == "prores_ks" else "yuv420p"
        args.extend(["-pix_fmt", output_pixel_format])
        if config.container in {"mp4", "mov"}:
            args.extend(["-movflags", "+faststart"])
        args.extend([str(partial_path)])
        return args

    def run(
        self,
        config: ProcessingConfig,
        on_progress: ProgressCallback,
        *,
        device_index: int = 0,
        fp16: bool = True,
        rife_scale: float | None = None,
        upscale_tile_size: int = 0,
        model_root: Path | None = None,
    ) -> Path:
        errors = validate_config(config)
        if errors:
            raise AIVideoExecutionError("Invalid job configuration: " + "; ".join(errors))
        if not self.ffmpeg_path:
            raise AIVideoExecutionError("FFmpeg was not found on PATH in this development build.")
        tools = discover_ffmpeg()
        if not tools.available:
            raise AIVideoExecutionError(tools.error or "FFmpeg/FFprobe are unavailable.")
        if config.video_codec.endswith("_nvenc") and config.video_codec not in tools.nvenc_encoders:
            raise AIVideoExecutionError(
                f"The installed FFmpeg build does not expose {config.video_codec}. "
                "Choose a supported NVENC codec or a software encoder."
            )

        metadata = probe_media(config.input_path)
        video = next(
            (stream for stream in metadata.get("streams", []) if stream.get("codec_type") == "video"),
            None,
        )
        if video is None:
            raise AIVideoExecutionError("The selected file does not contain a video stream.")
        width, height = int(video.get("width", 0)), int(video.get("height", 0))
        if width < 1 or height < 1:
            raise AIVideoExecutionError("FFprobe returned invalid source video dimensions.")

        source_rate = _rate_from_stream(video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/0")
        nominal_rate = _rate_from_stream(video.get("r_frame_rate") or video.get("avg_frame_rate") or "0/0")
        if abs(float(source_rate - nominal_rate)) > 0.1:
            raise AIVideoExecutionError(
                "Variable-frame-rate input is not supported in this build. "
                "Create a constant-frame-rate intermediate first; VFR timestamp-preserving processing is planned."
            )
        target_rate = as_rate(config.target_fps)
        # Prefer the video-stream duration. The container duration can be longer
        # when an audio stream has encoder delay or trailing samples.
        duration_candidates = (video.get("duration"), metadata.get("format", {}).get("duration"))
        duration = None
        for duration_value in duration_candidates:
            try:
                parsed_duration = float(duration_value)
            except (TypeError, ValueError):
                continue
            if parsed_duration > 0:
                duration = parsed_duration
                break
        if duration is None:
            frame_count_text = video.get("nb_frames")
            if not frame_count_text or frame_count_text == "N/A":
                raise AIVideoExecutionError("Could not determine source duration or frame count.")
            duration = int(frame_count_text) / float(source_rate)
        if duration <= 0:
            raise AIVideoExecutionError("Source video duration is invalid.")
        target_count = max(1, round(duration * float(target_rate)))

        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        if config.output_path.exists() and not config.overwrite:
            raise AIVideoExecutionError("Output already exists. Choose another filename or enable overwrite.")
        partial_path = config.output_path.with_name(
            f"{config.output_path.stem}.partial{config.output_path.suffix}"
        )
        partial_path.unlink(missing_ok=True)
        self._cancel.clear()

        decoder_command = [
            self.ffmpeg_path, "-hide_banner", "-loglevel", "error",
            "-i", str(config.input_path), "-map", "0:v:0", "-an", "-sn", "-dn",
            "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
        ]

        on_progress(0.0, "Loading CUDA AI models (first run downloads model weights)")
        needs_interpolation = config.interpolate and target_rate != source_rate
        needs_super_resolution = config.upscale and (
            config.output_width > width or config.output_height > height
        )
        scale_x = config.output_width / width
        scale_y = config.output_height / height
        # Use the x2 model for <=2x targets (notably 4K -> 8K), avoiding a
        # wasteful full x4 intermediate. Use x4 only when a larger native
        # reconstruction is useful; final resizing preserves the source aspect ratio.
        upscale_model_scale = 2 if max(scale_x, scale_y) <= 2.0 else 4
        with (
            tempfile.TemporaryFile(mode="w+b") as decoder_log,
            tempfile.TemporaryFile(mode="w+b") as encoder_log,
        ):
            output_count = 0
            input_count = 0
            previous_frame = None
            frame_bytes = width * height * 3
            pipeline = None

            try:
                def report_model_progress(percent: float, stage: str) -> None:
                    # Reserve the first 5% for checkpoint acquisition. If both
                    # models are required, allocate half to each so the display is
                    # monotonic when the second download starts.
                    if needs_interpolation and needs_super_resolution:
                        offset = 2.5 if "RealESRGAN" in stage else 0.0
                        model_percent = offset + min(100.0, max(0.0, percent)) * 0.025
                    else:
                        model_percent = min(5.0, max(0.0, percent * 0.05))
                    on_progress(model_percent, stage)

                pipeline = AIFramePipeline(
                    interpolate=needs_interpolation,
                    upscale=needs_super_resolution,
                    output_size=(config.output_width, config.output_height),
                    device_index=device_index,
                    fp16=fp16,
                    rife_scale=rife_scale,
                    upscale_tile_size=upscale_tile_size,
                    upscale_model_scale=upscale_model_scale,
                    model_root=model_root,
                    progress=report_model_progress,
                )
                on_progress(5.0, "CUDA models ready · starting video processing")
                self._encoder = subprocess.Popen(
                    self._encoder_command(config, width, height, target_rate, partial_path),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=encoder_log,
                    bufsize=0,
                )
                self._decoder = subprocess.Popen(
                    decoder_command,
                    stdout=subprocess.PIPE,
                    stderr=decoder_log,
                    bufsize=0,
                )
                if self._decoder.stdout is None or self._encoder.stdin is None:
                    raise AIVideoExecutionError("Could not initialize FFmpeg raw-frame pipes.")

                def write_output(frame) -> None:
                    nonlocal output_count
                    if self._cancel.is_set():
                        raise AIVideoExecutionError("Processing cancelled by user.")
                    if frame.shape != (config.output_height, config.output_width, 3):
                        raise AIVideoExecutionError(
                            f"AI pipeline returned invalid frame shape {frame.shape}; expected "
                            f"{(config.output_height, config.output_width, 3)}."
                        )
                    try:
                        _write_all(self._encoder.stdin, frame.tobytes())
                    except (BrokenPipeError, OSError) as exc:
                        raise AIVideoExecutionError("FFmpeg encoder closed its input pipe unexpectedly.") from exc
                    output_count += 1
                    on_progress(
                        min(99.0, 5.0 + output_count * 94.0 / target_count),
                        f"AI processing · frame {output_count:,}/{target_count:,}",
                    )

                while True:
                    raw = _read_exact(self._decoder.stdout, frame_bytes)
                    if raw is None:
                        break
                    if self._cancel.is_set():
                        raise AIVideoExecutionError("Processing cancelled by user.")
                    frame = np_from_rgb_bytes(raw, width, height)

                    if previous_frame is None:
                        write_output(pipeline.prepare_source_frame(frame))
                        previous_frame = frame
                        input_count = 1
                        continue

                    start_time = Fraction(input_count - 1, 1) / source_rate
                    end_time = Fraction(input_count, 1) / source_rate
                    while output_count < target_count:
                        target_time = Fraction(output_count, 1) / target_rate
                        if target_time >= end_time:
                            break
                        if target_time <= start_time:
                            raise AIVideoExecutionError("Frame timestamp scheduler moved backwards.")
                        alpha = float((target_time - start_time) / (end_time - start_time))
                        write_output(pipeline.process_pair(previous_frame, frame, alpha))

                    # Exact source-frame timestamps should use the original decoded frame.
                    if output_count < target_count:
                        target_time = Fraction(output_count, 1) / target_rate
                        if target_time == end_time:
                            write_output(pipeline.prepare_source_frame(frame))

                    previous_frame = frame
                    input_count += 1

                if input_count == 0 or previous_frame is None:
                    raise AIVideoExecutionError("FFmpeg decoded zero video frames.")

                # If duration metadata extends beyond the final presentation timestamp,
                # hold the final source frame to preserve the requested output cadence.
                while output_count < target_count:
                    write_output(pipeline.prepare_source_frame(previous_frame))

                self._encoder.stdin.close()
                self._encoder.stdin = None
                decoder_code = self._decoder.wait()
                encoder_code = self._encoder.wait()
                if decoder_code != 0:
                    decoder_log.seek(0)
                    detail = decoder_log.read()[-6000:].decode(errors="replace")
                    raise AIVideoExecutionError(detail or f"FFmpeg decoder exited with code {decoder_code}.")
                if encoder_code != 0:
                    encoder_log.seek(0)
                    detail = encoder_log.read()[-6000:].decode(errors="replace")
                    raise AIVideoExecutionError(detail or f"FFmpeg encoder exited with code {encoder_code}.")
                if not partial_path.is_file() or partial_path.stat().st_size == 0:
                    raise AIVideoExecutionError("FFmpeg did not produce a non-empty output file.")

                output_metadata = probe_media(partial_path)
                out_video = next(
                    (stream for stream in output_metadata.get("streams", []) if stream.get("codec_type") == "video"),
                    None,
                )
                if out_video is None:
                    raise AIVideoExecutionError("Output verification failed: no video stream found.")
                if int(out_video.get("width", 0)) != config.output_width or int(out_video.get("height", 0)) != config.output_height:
                    raise AIVideoExecutionError("Output verification failed: unexpected output resolution.")
                source_audio_streams = [
                    stream for stream in metadata.get("streams", []) if stream.get("codec_type") == "audio"
                ]
                output_audio_streams = [
                    stream for stream in output_metadata.get("streams", []) if stream.get("codec_type") == "audio"
                ]
                if len(output_audio_streams) != len(source_audio_streams):
                    raise AIVideoExecutionError(
                        "Output verification failed: the number of audio streams differs from the source."
                    )
                if config.resample_audio:
                    for audio_stream in output_audio_streams:
                        if int(audio_stream.get("sample_rate", 0)) != config.audio_sample_rate:
                            raise AIVideoExecutionError(
                                "Output verification failed: unexpected audio sample rate."
                            )

                output_rate_value = out_video.get("avg_frame_rate") or out_video.get("r_frame_rate")
                output_rate = _rate_from_stream(output_rate_value or "0/0")
                if output_rate != target_rate:
                    raise AIVideoExecutionError(
                        f"Output verification failed: expected {target_rate} FPS, got {output_rate} FPS."
                    )
                output_frame_count = out_video.get("nb_frames")
                if output_frame_count not in (None, "N/A") and int(output_frame_count) != target_count:
                    raise AIVideoExecutionError(
                        f"Output verification failed: expected {target_count} frames, got {output_frame_count}."
                    )
                # Verify the video stream duration, not the container duration: a
                # source audio stream may legitimately be slightly longer than video.
                output_video_duration = out_video.get("duration")
                if output_video_duration is not None:
                    try:
                        actual_video_duration = float(output_video_duration)
                    except (TypeError, ValueError):
                        actual_video_duration = None
                    if actual_video_duration is not None and abs(actual_video_duration - duration) > max(
                        0.05, 1.5 / float(target_rate)
                    ):
                        raise AIVideoExecutionError(
                            "Output verification failed: video-stream duration differs from the source by more than 1.5 frames."
                        )

                if config.output_path.exists() and config.overwrite:
                    config.output_path.unlink()
                partial_path.replace(config.output_path)
                on_progress(100.0, "AI processing complete · output verified")
                return config.output_path
            except Exception:
                partial_path.unlink(missing_ok=True)
                for process in (self._decoder, self._encoder):
                    if process is not None and process.poll() is None:
                        process.kill()
                raise
            finally:
                for process in (self._decoder, self._encoder):
                    if process is not None and process.poll() is None:
                        process.kill()
                self._decoder = None
                self._encoder = None
                if pipeline is not None:
                    pipeline.close()


def np_from_rgb_bytes(raw: bytes, width: int, height: int):
    import numpy as np

    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
