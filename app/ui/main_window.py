from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.config import ProcessingConfig, validate_config
from app.core.pipeline import PipelineNotReadyError, validate_ai_pipeline
from app.engines.availability import realesrgan_status, rife_status, torch_cuda_status
from app.hardware.gpu import detect_gpus
from app.hardware.system import system_metrics
from app.media.ffmpeg import discover_ffmpeg, probe_media
from app.paths import log_dir, model_dir
from app.ui.theme import DARK_QSS
from app.version import (
    APP_AUTHOR,
    APP_HOMEPAGE,
    APP_NAME,
    APP_TAGLINE,
    NVIDIA_DRIVER_URL,
    WINDOW_TITLE,
    display_version,
)


class MainWindow(QMainWindow):
    def __init__(self, run_startup_checks: bool = True) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1280, 820)
        self.setMinimumSize(1020, 700)
        self.source_path: Path | None = None
        self.media_duration = 0.0
        self._worker_thread: QThread | None = None
        self._worker = None
        self._build_ui()
        self.setStyleSheet(DARK_QSS)
        self._refresh_status()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh_status)
        self.timer.start(2500)
        self._check_thread: QThread | None = None
        self._check_worker = None
        if run_startup_checks:
            QTimer.singleShot(0, self._start_runtime_check)

    def _panel(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        panel = QFrame()
        panel.setObjectName("Panel")
        layout = QVBoxLayout(panel)
        heading = QLabel(title)
        heading.setStyleSheet("font-size: 13pt; font-weight: 650;")
        layout.addWidget(heading)
        return panel, layout

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QHBoxLayout(root)
        outer.setContentsMargins(18, 18, 18, 18)
        outer.setSpacing(16)
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        side = QVBoxLayout(sidebar)
        brand = QLabel("▰  AI UPSCALER\n     STUDIO")
        brand.setStyleSheet("font-size: 15pt; font-weight: 800; color: #A8E34E; padding: 10px;")
        side.addWidget(brand)
        tagline = QLabel(f"{APP_TAGLINE}\n{display_version()}")
        tagline.setStyleSheet("color: #9AA4AD; padding: 0 10px 8px 10px;")
        side.addWidget(tagline)
        self.nav = QListWidget()
        self.nav.addItems(["Dashboard", "Process Video", "Queue", "Settings"])
        self.nav.setFixedWidth(205)
        side.addWidget(self.nav)
        side.addStretch()
        self.driver_label = QLabel("Detecting GPU…")
        self.driver_label.setWordWrap(True)
        self.driver_label.setStyleSheet("color: #9AA4AD; padding: 8px;")
        side.addWidget(self.driver_label)
        outer.addWidget(sidebar)
        self.pages = QStackedWidget()
        self.pages.addWidget(self._dashboard_page())
        self.pages.addWidget(self._process_page())
        self.pages.addWidget(self._simple_page("Processing queue", "Persistent multi-job queue is not yet implemented."))
        self.pages.addWidget(self._settings_page())
        outer.addWidget(self.pages, 1)
        self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.nav.setCurrentRow(0)
        self.setCentralWidget(root)

    def _dashboard_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel("Dashboard")
        title.setObjectName("Title")
        layout.addWidget(title)
        sub = QLabel("Local video processing · NVIDIA hardware diagnostics")
        sub.setObjectName("Subtitle")
        layout.addWidget(sub)
        grid = QGridLayout()
        self.metric_labels: dict[str, QLabel] = {}
        for index, (key, label_text) in enumerate([
            ("gpu", "GPU utilization"), ("vram", "VRAM"), ("cpu", "CPU utilization"),
            ("ram", "System memory"), ("temp", "GPU temperature"), ("nvenc", "NVENC encoders"),
        ]):
            panel, box = self._panel(label_text)
            value = QLabel("—")
            value.setStyleSheet("font-size: 21pt; font-weight: 700; color: #A8E34E;")
            box.addWidget(value)
            self.metric_labels[key] = value
            grid.addWidget(panel, index // 2, index % 2)
        layout.addLayout(grid)
        panel, box = self._panel("Processing engines and system readiness")
        self.readiness = QLabel("Checking dependencies…")
        self.readiness.setWordWrap(True)
        box.addWidget(self.readiness)
        layout.addWidget(panel)
        layout.addStretch()
        return page

    def _process_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel("Process video")
        title.setObjectName("Title")
        layout.addWidget(title)
        sub = QLabel("Select a source, target frame rate, output resolution and audio sample rate.")
        sub.setObjectName("Subtitle")
        layout.addWidget(sub)
        panel, box = self._panel("Source video")
        row = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Choose a video file…")
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse)
        row.addWidget(self.input_edit, 1)
        row.addWidget(browse)
        box.addLayout(row)
        self.media_info = QLabel("No media selected")
        self.media_info.setStyleSheet("color: #9AA4AD;")
        box.addWidget(self.media_info)
        layout.addWidget(panel)
        panel, box = self._panel("Output configuration")
        form = QGridLayout()
        form.addWidget(QLabel("Target FPS"), 0, 0)
        self.fps = QDoubleSpinBox()
        self.fps.setRange(1.0, 240.0)
        self.fps.setDecimals(3)
        self.fps.setValue(60.0)
        form.addWidget(self.fps, 0, 1)
        form.addWidget(QLabel("Resolution"), 0, 2)
        self.resolution = QComboBox()
        self.resolution.addItems(["1280 × 720", "1920 × 1080", "2560 × 1440", "3840 × 2160", "5120 × 2880", "7680 × 4320"])
        self.resolution.setCurrentText("3840 × 2160")
        form.addWidget(self.resolution, 0, 3)
        form.addWidget(QLabel("Audio sample rate"), 1, 0)
        self.sample_rate = QComboBox()
        self.sample_rate.addItems(["44100", "48000", "96000"])
        self.sample_rate.setCurrentText("48000")
        form.addWidget(self.sample_rate, 1, 1)
        form.addWidget(QLabel("Container"), 1, 2)
        self.container = QComboBox()
        self.container.addItems(["mp4", "mkv", "mov", "webm"])
        form.addWidget(self.container, 1, 3)
        form.addWidget(QLabel("Video codec"), 2, 0)
        self.codec = QComboBox()
        form.addWidget(self.codec, 2, 1)
        self.container.currentTextChanged.connect(self._set_codecs)
        self._set_codecs()
        self.interpolate = QPushButton("AI Interpolation: ON")
        self.interpolate.setCheckable(True)
        self.interpolate.setChecked(True)
        self.interpolate.toggled.connect(lambda v: self.interpolate.setText(f"AI Interpolation: {'ON' if v else 'OFF'}"))
        form.addWidget(self.interpolate, 2, 2)
        self.upscale = QPushButton("AI Upscaling: ON")
        self.upscale.setCheckable(True)
        self.upscale.setChecked(True)
        self.upscale.toggled.connect(lambda v: self.upscale.setText(f"AI Upscaling: {'ON' if v else 'OFF'}"))
        form.addWidget(self.upscale, 2, 3)
        form.addWidget(QLabel("NVIDIA GPU"), 3, 0)
        self.gpu_selector = QComboBox()
        self._populate_gpu_selector()
        form.addWidget(self.gpu_selector, 3, 1)
        self.fp16_button = QPushButton("FP16 inference: ON")
        self.fp16_button.setCheckable(True)
        self.fp16_button.setChecked(True)
        self.fp16_button.toggled.connect(
            lambda value: self.fp16_button.setText(f"FP16 inference: {'ON' if value else 'OFF'}")
        )
        form.addWidget(self.fp16_button, 3, 2)
        box.addLayout(form)
        outrow = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Output path…")
        outbrowse = QPushButton("Save as")
        outbrowse.clicked.connect(self._choose_output)
        outrow.addWidget(self.output_edit, 1)
        outrow.addWidget(outbrowse)
        box.addLayout(outrow)
        self.start_button = QPushButton("Start AI processing")
        self.start_button.setObjectName("Primary")
        self.start_button.clicked.connect(self._validate_job)
        box.addWidget(self.start_button)
        self.basic_button = QPushButton("Run standard FFmpeg export (no AI)")
        self.basic_button.clicked.connect(self._run_basic_export)
        box.addWidget(self.basic_button)
        self.job_message = QLabel("No job submitted.")
        self.job_message.setWordWrap(True)
        box.addWidget(self.job_message)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        box.addWidget(self.progress)
        layout.addWidget(panel)
        layout.addStretch()
        return page

    def _simple_page(self, title_text: str, message: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel(title_text)
        title.setObjectName("Title")
        layout.addWidget(title)
        label = QLabel(message)
        label.setWordWrap(True)
        label.setObjectName("Subtitle")
        layout.addWidget(label)
        layout.addStretch()
        return page

    def _settings_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel("Settings")
        title.setObjectName("Title")
        layout.addWidget(title)
        sub = QLabel("Runtime diagnostics and storage locations")
        sub.setObjectName("Subtitle")
        layout.addWidget(sub)
        panel, box = self._panel("Environment and hardware check")
        self.diagnostics = QLabel("Running startup checks…")
        self.diagnostics.setWordWrap(True)
        self.diagnostics.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        box.addWidget(self.diagnostics)
        recheck = QPushButton("Run checks again")
        recheck.clicked.connect(self._start_runtime_check)
        box.addWidget(recheck)
        layout.addWidget(panel)
        panel, box = self._panel("Storage")
        storage = QLabel(
            f"Model cache: {model_dir()}\nLogs: {log_dir()}\n"
            "Model checkpoints download automatically on first use and are reused afterwards."
        )
        storage.setWordWrap(True)
        storage.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        box.addWidget(storage)
        layout.addWidget(panel)
        panel, box = self._panel("About")
        about = QLabel(
            f"{APP_NAME} — {APP_TAGLINE}<br>{display_version()}<br>"
            f"Created and developed by {APP_AUTHOR}<br>"
            f'<a href="{APP_HOMEPAGE}">{APP_HOMEPAGE}</a><br>'
            "Independent project; not affiliated with or endorsed by NVIDIA Corporation."
        )
        about.setTextFormat(Qt.TextFormat.RichText)
        about.setOpenExternalLinks(True)
        about.setWordWrap(True)
        box.addWidget(about)
        layout.addWidget(panel)
        layout.addStretch()
        return page

    def _start_runtime_check(self) -> None:
        if self._check_thread is not None:
            return
        from app.ui.workers import RuntimeCheckWorker

        self.diagnostics.setText("Running startup checks…")
        self._check_thread = QThread(self)
        self._check_worker = RuntimeCheckWorker()
        self._check_worker.moveToThread(self._check_thread)
        self._check_thread.started.connect(self._check_worker.run)
        self._check_worker.finished.connect(self._on_runtime_check)
        self._check_worker.finished.connect(self._check_thread.quit)
        self._check_thread.finished.connect(self._on_check_thread_finished)
        self._check_thread.start()

    def closeEvent(self, event) -> None:
        if self._worker_thread is not None and self._worker_thread.isRunning():
            answer = QMessageBox.question(
                self, "Processing in progress",
                "A video is still being processed. Cancel the job and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self.shutdown()
        super().closeEvent(event)

    def shutdown(self) -> None:
        """Stop timers and join worker threads before Qt and CUDA are torn down.

        Destroying a running QThread aborts the process, so every background
        thread is cancelled and joined here.
        """
        self.timer.stop()
        worker, worker_thread = self._worker, self._worker_thread
        if worker_thread is not None and worker_thread.isRunning():
            if worker is not None and hasattr(worker, "cancel"):
                worker.cancel()  # terminates FFmpeg; the runner then raises and returns
            worker_thread.quit()
            worker_thread.wait(30_000)
        check_thread = self._check_thread
        if check_thread is not None and check_thread.isRunning():
            check_thread.quit()
            check_thread.wait(5_000)

    def _on_check_thread_finished(self) -> None:
        self._check_worker = None
        self._check_thread = None

    def _on_runtime_check(self, report) -> None:
        from app.hardware.runtime_check import ERROR

        self.diagnostics.setText(report.as_text())
        self._refresh_status()
        if report.level == ERROR and not getattr(self, "_runtime_warning_shown", False):
            self._runtime_warning_shown = True
            problems = "".join(
                f"<li>{item.name}: {item.detail}</li>" for item in report.problems() if item.status == ERROR
            )
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("AI processing is not available yet")
            box.setTextFormat(Qt.TextFormat.RichText)
            box.setText(
                "<p>AI Upscaler Studio needs a compatible NVIDIA GPU and NVIDIA display driver for "
                f"AI processing.</p><ul>{problems}</ul>"
                f'<p>Download the latest driver from NVIDIA: <a href="{NVIDIA_DRIVER_URL}">'
                f"{NVIDIA_DRIVER_URL}</a></p><p>The standard FFmpeg export remains available. "
                "Details are on the Settings page.</p>"
            )
            box.open()

    def _set_codecs(self) -> None:
        mapping = {
            "mp4": ["h264_nvenc", "hevc_nvenc", "av1_nvenc", "libx264", "libx265"],
            "mkv": ["h264_nvenc", "hevc_nvenc", "av1_nvenc", "libx264", "libx265", "libvpx-vp9"],
            "mov": ["h264_nvenc", "hevc_nvenc", "prores_ks", "libx264", "libx265"],
            "webm": ["libvpx-vp9", "libaom-av1", "av1_nvenc"],
        }
        previous = self.codec.currentText()
        self.codec.clear()
        self.codec.addItems(mapping[self.container.currentText()])
        index = self.codec.findText(previous)
        if index >= 0:
            self.codec.setCurrentIndex(index)

    def _browse(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Select video", "", "Media files (*.*)")
        if not filename:
            return
        self.source_path = Path(filename)
        self.input_edit.setText(filename)
        try:
            data = probe_media(self.source_path)
            self.media_duration = float(data.get("format", {}).get("duration", 0.0) or 0.0)
            streams = data.get("streams", [])
            video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
            if video is None:
                raise ValueError("Selected file has no video stream.")
            self.media_info.setText(
                f"{video.get('codec_name', 'unknown')} · {video.get('width', '?')} × "
                f"{video.get('height', '?')} · {video.get('r_frame_rate', '?')} FPS · "
                f"{self.media_duration:.2f} seconds"
            )
        except (OSError, RuntimeError, ValueError) as exc:
            self.media_info.setText(f"Media probe failed: {exc}")

    def _choose_output(self) -> None:
        extension = self.container.currentText()
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save output video", f"enhanced.{extension}",
            "MP4 (*.mp4);;MKV (*.mkv);;MOV (*.mov);;WebM (*.webm)",
        )
        if filename:
            self.output_edit.setText(filename)

    def _populate_gpu_selector(self) -> None:
        """List PyTorch-visible NVIDIA devices, not merely display adapters."""
        self.gpu_selector.clear()
        try:
            import torch

            if torch.cuda.is_available():
                for index in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(index)
                    vram_gb = props.total_memory / (1024 ** 3)
                    self.gpu_selector.addItem(
                        f"{torch.cuda.get_device_name(index)} · {vram_gb:.1f} GB",
                        index,
                    )
        except (ImportError, OSError, RuntimeError) as exc:
            self.gpu_selector.addItem(f"CUDA unavailable: {exc}", 0)
        if self.gpu_selector.count() == 0:
            self.gpu_selector.addItem("No CUDA GPU available", 0)

    def _make_config(self, ai_enabled: bool) -> ProcessingConfig | None:
        if not self.source_path:
            QMessageBox.warning(self, "Missing input", "Choose a source video first.")
            return None
        output_text = self.output_edit.text().strip()
        if not output_text:
            QMessageBox.warning(self, "Missing output", "Choose an output path.")
            return None
        width, height = (int(part.strip()) for part in self.resolution.currentText().split("×"))
        return ProcessingConfig(
            input_path=self.source_path, output_path=Path(output_text),
            target_fps=self.fps.value(), output_width=width, output_height=height,
            audio_sample_rate=int(self.sample_rate.currentText()),
            container=self.container.currentText(), video_codec=self.codec.currentText(),
            interpolate=ai_enabled and self.interpolate.isChecked(),
            upscale=ai_enabled and self.upscale.isChecked(),
            gpu_index=int(self.gpu_selector.currentData() or 0),
            fp16=self.fp16_button.isChecked(),
        )

    def _validate_job(self) -> None:
        config = self._make_config(ai_enabled=True)
        if config is None:
            return
        errors = validate_config(config)
        if errors:
            QMessageBox.warning(self, "Invalid configuration", "\n".join(errors))
            return
        try:
            validate_ai_pipeline(config)
        except PipelineNotReadyError as exc:
            self.job_message.setText(str(exc))
            QMessageBox.warning(self, "AI engines unavailable", str(exc))
            return
        self._start_ai_worker(config)

    def _start_ai_worker(self, config: ProcessingConfig) -> None:
        from app.ui.workers import AIProcessingWorker

        self.start_button.setEnabled(False)
        self.basic_button.setEnabled(False)
        self.progress.setValue(0)
        self.job_message.setText("Preparing CUDA AI pipeline; first run downloads model weights…")
        self._worker_thread = QThread(self)
        self._worker = AIProcessingWorker(config)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.succeeded.connect(self._on_success)
        self._worker.failed.connect(self._on_failure)
        self._worker.succeeded.connect(self._worker_thread.quit)
        self._worker.failed.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._on_worker_finished)
        self._worker_thread.start()

    def _run_basic_export(self) -> None:
        """Run a clearly labelled conventional FFmpeg conversion (not AI)."""
        config = self._make_config(ai_enabled=False)
        if config is None:
            return
        errors = validate_config(config)
        if errors:
            QMessageBox.warning(self, "Invalid configuration", "\n".join(errors))
            return
        answer = QMessageBox.warning(
            self, "Standard FFmpeg export",
            "This export uses FFmpeg FPS conversion and Lanczos scaling. It is NOT AI motion "
            "interpolation or AI super-resolution. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._start_worker(config)

    def _start_worker(self, config: ProcessingConfig) -> None:
        from app.ui.workers import TranscodeWorker
        self.start_button.setEnabled(False)
        self.basic_button.setEnabled(False)
        self.progress.setValue(0)
        self.job_message.setText("Starting standard FFmpeg export…")
        self._worker_thread = QThread(self)
        self._worker = TranscodeWorker(config, self.media_duration)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.succeeded.connect(self._on_success)
        self._worker.failed.connect(self._on_failure)
        self._worker.succeeded.connect(self._worker_thread.quit)
        self._worker.failed.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._on_worker_finished)
        self._worker_thread.start()

    def _on_progress(self, percent: float, stage: str) -> None:
        self.progress.setValue(round(percent))
        self.job_message.setText(f"{stage} · {percent:.1f}%")

    def _on_success(self, output_path: str) -> None:
        self.progress.setValue(100)
        self.job_message.setText(f"Export completed: {output_path}")
        QMessageBox.information(self, "Export complete", output_path)

    def _on_failure(self, error: str) -> None:
        self.job_message.setText(f"Export failed: {error}")
        QMessageBox.critical(self, "Export failed", error)

    def _on_worker_finished(self) -> None:
        self.start_button.setEnabled(True)
        self.basic_button.setEnabled(True)
        self._worker = None
        self._worker_thread = None

    def _refresh_status(self) -> None:
        gpu = detect_gpus()[0]
        system = system_metrics()
        self.metric_labels["gpu"].setText(f"{gpu.utilization_percent}%" if gpu.utilization_percent is not None else "—")
        self.metric_labels["vram"].setText(
            f"{gpu.vram_total_mb - gpu.vram_free_mb:,} / {gpu.vram_total_mb:,} MB"
            if gpu.vram_total_mb is not None and gpu.vram_free_mb is not None else "—"
        )
        self.metric_labels["cpu"].setText(f"{system['cpu_percent']:.0f}%")
        self.metric_labels["ram"].setText(f"{system['ram_used_gb']:.1f} / {system['ram_total_gb']:.1f} GB")
        self.metric_labels["temp"].setText(f"{gpu.temperature_c} °C" if gpu.temperature_c is not None else "—")
        ffmpeg = discover_ffmpeg()
        self.metric_labels["nvenc"].setText(", ".join(ffmpeg.nvenc_encoders) or "Not detected")
        self.driver_label.setText(
            f"{gpu.name}\nDriver: {gpu.driver_version or 'not detected'}"
            if gpu.available else (gpu.warning or "NVIDIA GPU unavailable")
        )
        cuda, rife, sr = torch_cuda_status(), rife_status(), realesrgan_status()
        self.readiness.setText(
            f"GPU: {'detected' if gpu.available else 'not detected'} · "
            f"FFmpeg: {'available' if ffmpeg.available else 'missing'} · "
            f"{cuda.name}: {'available' if cuda.available else 'unavailable'} · "
            f"{rife.name}: {'available' if rife.available else 'not integrated'} · "
            f"{sr.name}: {'available' if sr.available else 'not integrated'}"
        )
