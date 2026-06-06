"""Project-wide logging configuration.

A single :func:`get_logger` factory produces correctly configured logger
instances that simultaneously emit to ``stdout`` and to a rotating log file
under :data:`config.settings.LOGS_DIR`. The factory is idempotent: importing
it multiple times will never duplicate handlers.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

from config import settings

_DEFAULT_MAX_BYTES: int = 5 * 1024 * 1024
_DEFAULT_BACKUP_COUNT: int = 3
_CONFIGURED: bool = False


def _configure_root_logger() -> None:
    """Attach console and rotating-file handlers to the root logger.

    The configuration is applied exactly once per Python process; subsequent
    invocations are a no-op so repeated module imports remain cheap and free
    of duplicated log lines.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(settings.LOG_LEVEL)

    formatter = logging.Formatter(
        fmt=settings.LOG_FORMAT,
        datefmt=settings.LOG_DATEFMT,
    )

    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(settings.LOG_LEVEL)
        root_logger.addHandler(console_handler)

    try:
        settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=settings.LOG_FILE,
            maxBytes=_DEFAULT_MAX_BYTES,
            backupCount=_DEFAULT_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(settings.LOG_LEVEL)
        root_logger.addHandler(file_handler)
    except OSError as exc:
        root_logger.warning(
            "Could not attach file handler at %s (%s); falling back to console only.",
            settings.LOG_FILE,
            exc,
        )

    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger ready for production use.

    Args:
        name: Dotted logger name. ``None`` returns the root logger. Modules
            should usually pass ``__name__`` so log records carry the calling
            module path.

    Returns:
        A :class:`logging.Logger` instance with console and rotating file
        handlers wired up.
    """
    _configure_root_logger()
    return logging.getLogger(name)
