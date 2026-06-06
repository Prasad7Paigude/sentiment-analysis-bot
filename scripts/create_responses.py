"""Regenerate the pickled chatbot response bank.

The response bank is consumed by :class:`src.response_generator.ChatbotResponder`.
Re-running this script overwrites ``models/responses.pkl`` with the latest
catalogue declared in :data:`config.settings.DEFAULT_RESPONSES`.

Usage::

    python -m scripts.create_responses
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from utils.artifact_io import ArtifactLoadError, save_pickle
from utils.logger import get_logger

logger = get_logger(__name__)


def regenerate_response_bank() -> Path:
    """Persist :data:`config.settings.DEFAULT_RESPONSES` to disk.

    Returns:
        The destination path the file was written to.

    Raises:
        ArtifactLoadError: If the pickle file cannot be written.
    """
    logger.info(
        "Regenerating chatbot response bank with %d sentiment classes.",
        len(settings.DEFAULT_RESPONSES),
    )
    destination = save_pickle(dict(settings.DEFAULT_RESPONSES), settings.RESPONSES_PATH)
    logger.info("Responses saved successfully to %s", destination)
    return destination


def main() -> int:
    """Entry point used by ``python -m scripts.create_responses``.

    Returns:
        Process exit code (``0`` on success, ``1`` on any failure).
    """
    try:
        regenerate_response_bank()
    except ArtifactLoadError as exc:
        logger.error("Could not write response bank: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
