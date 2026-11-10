from __future__ import annotations

from app.core.config import ProcessingConfig


def build_ffmpeg_transcode_command(ffmpeg: str, config: ProcessingConfig) -> list[str]:
    """Build conventional FPS conversion and Lanczos scaling; these are NOT AI effects."""
    filters = [f"fps={config.target_fps:g}"]
    filters.append(
        f"scale={config.output_width}:{config.output_height}:flags=lanczos:force_original_aspect_ratio=decrease"
    )
    filters.append(f"pad={config.output_width}:{config.output_height}:(ow-iw)/2:(oh-ih)/2")
    args = [ffmpeg, "-hide_banner", "-y" if config.overwrite else "-n", "-i", str(config.input_path),
            "-vf", ",".join(filters)]
    if config.video_codec.endswith("_nvenc"):
        args.extend(["-c:v", config.video_codec, "-preset", "p6", "-tune", "hq",
                     "-rc", "vbr", "-cq", "19", "-b:v", "0"])
    else:
        args.extend(["-c:v", config.video_codec])
    args.extend(["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                 "-ar", str(config.audio_sample_rate)])
    args.extend(["-af", f"aresample={config.audio_sample_rate}", "-progress", "pipe:1",
                 "-nostats", str(config.output_path)])
    return args
