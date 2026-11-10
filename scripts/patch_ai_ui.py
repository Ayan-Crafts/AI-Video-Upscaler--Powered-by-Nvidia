from __future__ import annotations

from pathlib import Path

path = Path("app/ui/main_window.py")
source = path.read_text(encoding="utf-8")

# Add GPU/precision controls to the Process Video page.
old_controls = '''        self.upscale.toggled.connect(lambda v: self.upscale.setText(f"AI Upscaling: {'ON' if v else 'OFF'}"))
        form.addWidget(self.upscale, 2, 3)
        box.addLayout(form)
'''
new_controls = '''        self.upscale.toggled.connect(lambda v: self.upscale.setText(f"AI Upscaling: {'ON' if v else 'OFF'}"))
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
'''
if "self.gpu_selector = QComboBox()" not in source:
    if old_controls not in source:
        raise SystemExit("Could not locate Process Video controls. No UI changes made.")
    source = source.replace(old_controls, new_controls, 1)

# Add the GPU population helper once.
helper_marker = "    def _make_config(self, ai_enabled: bool) -> ProcessingConfig | None:\n"
helper = '''    def _populate_gpu_selector(self) -> None:
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

'''
if "def _populate_gpu_selector(" not in source:
    if helper_marker not in source:
        raise SystemExit("Could not locate _make_config insertion point. No UI changes made.")
    source = source.replace(helper_marker, helper + helper_marker, 1)

# Pass selected device and precision into the job config.
old_config = '''            container=self.container.currentText(), video_codec=self.codec.currentText(),
            interpolate=ai_enabled and self.interpolate.isChecked(),
            upscale=ai_enabled and self.upscale.isChecked(),
        )
'''
new_config = '''            container=self.container.currentText(), video_codec=self.codec.currentText(),
            interpolate=ai_enabled and self.interpolate.isChecked(),
            upscale=ai_enabled and self.upscale.isChecked(),
            gpu_index=int(self.gpu_selector.currentData() or 0),
            fp16=self.fp16_button.isChecked(),
        )
'''
if "gpu_index=int(self.gpu_selector.currentData() or 0)" not in source:
    if old_config not in source:
        raise SystemExit("Could not locate ProcessingConfig creation. No GPU selection wiring made.")
    source = source.replace(old_config, new_config, 1)

# AI action must start the background worker after validation.
old_validation = '''        try:
            validate_ai_pipeline(config)
        except PipelineNotReadyError as exc:
            self.job_message.setText(str(exc))
            QMessageBox.warning(self, "AI engines unavailable", str(exc))
'''
new_validation = '''        try:
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
'''
if "def _start_ai_worker(" not in source:
    if old_validation not in source:
        raise SystemExit("Could not locate AI validation block. No AI worker wiring made.")
    source = source.replace(old_validation, new_validation, 1)

source = source.replace(
    'self.start_button = QPushButton("Validate AI processing job")',
    'self.start_button = QPushButton("Start AI processing")',
    1,
)
path.write_text(source, encoding="utf-8")
print("Patched Process Video UI with CUDA GPU selection, FP16 control and AI worker startup.")
