from __future__ import annotations

import argparse
import faulthandler
import logging
import logging.handlers
import os
import sys
from pathlib import Path

from app.paths import ensure_dir, icon_path, log_dir
from app.version import APP_NAME, APP_TAGLINE, display_version

logger = logging.getLogger(__name__)
_crash_log = None


def setup_logging() -> Path | None:
    """Log to the per-user log directory, and to stderr when a console exists."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    if sys.stderr is not None:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)
    log_file: Path | None = None
    try:
        log_file = ensure_dir(log_dir()) / "ai-upscaler-studio.log"
        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)
        # Native crashes (CUDA, Qt) bypass logging; record their Python stacks.
        global _crash_log
        _crash_log = (log_file.parent / "crash.log").open("a", encoding="utf-8")
        faulthandler.enable(file=_crash_log, all_threads=True)
    except OSError as exc:  # read-only home or similar; keep running
        log_file = None
        logger.warning("File logging disabled: %s", exc)

    def log_uncaught(exc_type, exc, traceback) -> None:
        logging.getLogger("app").critical("Unhandled exception", exc_info=(exc_type, exc, traceback))
        sys.__excepthook__(exc_type, exc, traceback)

    sys.excepthook = log_uncaught
    return log_file


def _hard_exit(code: int) -> None:
    """Flush logs and output, then end the process without interpreter finalization.

    With CUDA PyTorch and Qt loaded together, their C++ teardown order at
    interpreter exit is not deterministic and a native thread could abort the
    process (observed as exit code 0xC0000409 on Windows) after all user-visible
    work had already finished. All state is persisted before this point.
    """
    logger.info("Exiting with code %s", code)
    logging.shutdown()
    if _crash_log is not None:
        faulthandler.disable()
        _crash_log.close()
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            try:
                stream.flush()
            except (OSError, ValueError):
                pass
    os._exit(code)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ai-upscaler-studio", description=f"{APP_NAME} — {APP_TAGLINE}")
    parser.add_argument("--version", action="store_true", help="Print the version and exit.")
    parser.add_argument("--self-test", action="store_true",
                        help="Check the bundled runtime, Qt, FFmpeg and NVIDIA GPU, then exit.")
    parser.add_argument("--e2e", action="store_true",
                        help="With --self-test: download/load models and process a short test video.")
    parser.add_argument("--require-gpu", action="store_true",
                        help="With --self-test: fail unless CUDA inference is available.")
    parser.add_argument("--report", type=Path, default=None,
                        help="With --self-test: write the JSON report to this file.")
    # Qt consumes its own arguments (for example -platform); ignore unknown ones.
    args, _unknown = parser.parse_known_args(argv)
    return args


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    if args.version:
        if sys.stdout is not None:
            print(f"{APP_NAME} {display_version()}")
        return 0

    log_file = setup_logging()
    logger.info("Starting %s %s (log: %s)", APP_NAME, display_version(), log_file)

    if args.self_test:
        from app.selftest import run_self_test

        code = run_self_test(e2e=args.e2e, require_gpu=args.require_gpu, report_path=args.report)
        _hard_exit(code)

    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from app.ui.main_window import MainWindow

    application = QApplication([sys.argv[0], *argv])
    application.setApplicationName(APP_NAME)
    application.setOrganizationName(APP_NAME)
    application.setApplicationVersion(display_version())
    application.setDesktopFileName("ai-upscaler-studio")
    icon = icon_path()
    if icon.is_file():
        application.setWindowIcon(QIcon(str(icon)))
    window = MainWindow()
    window.show()
    exit_code = application.exec()
    # Orderly cleanup: cancel any job (terminates FFmpeg, removes the partial
    # output) and join background threads with bounded waits, then flush logs.
    window.shutdown()
    _hard_exit(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
