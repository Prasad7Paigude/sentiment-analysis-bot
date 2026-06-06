"""Re-run the end-to-end model training pipeline from the command line.

The pipeline mirrors what the original ``notebooks/model_training.ipynb`` did:

1. Load ``data/Reddit_Data.csv``.
2. Clean + tokenise the text.
3. Fit a TF-IDF vectorizer and persist it under ``models/``.
4. Train the dense classifier and persist it under ``models/``.
5. Print evaluation metrics.

Usage::

    python -m scripts.train
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from src.model_training import run_training_pipeline
from utils.logger import get_logger

logger = get_logger(__name__)


def main() -> int:
    """Execute the training pipeline and report status to stdout/logs.

    Returns:
        Process exit code (``0`` on success, ``1`` on any failure).
    """
    logger.info("Starting training pipeline (root=%s).", settings.PROJECT_ROOT)
    try:
        _, metrics = run_training_pipeline()
    except FileNotFoundError as exc:
        logger.error("Required input file missing: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Training pipeline failed: %s", exc)
        return 1

    logger.info("Training succeeded. Final test accuracy: %.4f", metrics["accuracy"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
