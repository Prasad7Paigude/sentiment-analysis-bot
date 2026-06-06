"""Centralised configuration for the Sentiment Analysis Chatbot.

All filesystem paths, model hyper-parameters, runtime constants and tunable
behavioural flags live in this module. Application code MUST import values
from here rather than hard-coding strings or magic numbers.

Environment variables (loaded transparently when ``python-dotenv`` is
available) can override every configurable value, which keeps the codebase
twelve-factor friendly and CI/CD ready.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Final, List, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _get_env(key: str, default: str) -> str:
    """Return the value of an environment variable, falling back to ``default``.

    Args:
        key: Name of the environment variable to read.
        default: Value returned when the variable is unset or empty.

    Returns:
        The resolved string value (never ``None``).
    """
    value = os.environ.get(key, "").strip()
    return value if value else default


def _get_env_int(key: str, default: int) -> int:
    """Return an integer-valued environment variable with a safe fallback.

    Args:
        key: Name of the environment variable.
        default: Fallback integer used when the variable is missing or invalid.

    Returns:
        The parsed integer, or ``default`` when parsing fails.
    """
    try:
        return int(_get_env(key, str(default)))
    except ValueError:
        return default


def _get_env_float(key: str, default: float) -> float:
    """Return a float-valued environment variable with a safe fallback.

    Args:
        key: Name of the environment variable.
        default: Fallback float used when the variable is missing or invalid.

    Returns:
        The parsed float, or ``default`` when parsing fails.
    """
    try:
        return float(_get_env(key, str(default)))
    except ValueError:
        return default


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

DATA_DIR: Final[Path] = Path(_get_env("DATA_DIR", str(PROJECT_ROOT / "data")))
MODELS_DIR: Final[Path] = Path(_get_env("MODELS_DIR", str(PROJECT_ROOT / "models")))
LOGS_DIR: Final[Path] = Path(_get_env("LOGS_DIR", str(PROJECT_ROOT / "logs")))
NOTEBOOKS_DIR: Final[Path] = PROJECT_ROOT / "notebooks"

RAW_DATASET_PATH: Final[Path] = DATA_DIR / _get_env(
    "RAW_DATASET_FILENAME", "Reddit_Data.csv"
)

SENTIMENT_MODEL_PATH: Final[Path] = MODELS_DIR / _get_env(
    "SENTIMENT_MODEL_FILENAME", "sentiment_analysis.h5"
)
TFIDF_VECTORIZER_PATH: Final[Path] = MODELS_DIR / _get_env(
    "TFIDF_VECTORIZER_FILENAME", "tfidf_vectorizer.pkl"
)
RESPONSES_PATH: Final[Path] = MODELS_DIR / _get_env(
    "RESPONSES_FILENAME", "responses.pkl"
)

TEXT_COLUMN: Final[str] = _get_env("TEXT_COLUMN", "text")
LABEL_COLUMN: Final[str] = _get_env("LABEL_COLUMN", "category")
CLEANED_TEXT_COLUMN: Final[str] = "cleaned_text"
TOKENS_COLUMN: Final[str] = "tokens"

TFIDF_MAX_FEATURES: Final[int] = _get_env_int("TFIDF_MAX_FEATURES", 5000)

TEST_SIZE: Final[float] = _get_env_float("TEST_SIZE", 0.2)
RANDOM_STATE: Final[int] = _get_env_int("RANDOM_STATE", 42)

NUM_CLASSES: Final[int] = 3
LABEL_OFFSET: Final[int] = 1
DENSE_UNITS_LAYER_1: Final[int] = _get_env_int("DENSE_UNITS_LAYER_1", 256)
DENSE_UNITS_LAYER_2: Final[int] = _get_env_int("DENSE_UNITS_LAYER_2", 128)
DROPOUT_RATE: Final[float] = _get_env_float("DROPOUT_RATE", 0.5)
HIDDEN_ACTIVATION: Final[str] = "relu"
OUTPUT_ACTIVATION: Final[str] = "softmax"
LOSS_FUNCTION: Final[str] = "categorical_crossentropy"
OPTIMIZER: Final[str] = "adam"
METRICS: Final[List[str]] = ["accuracy"]

EPOCHS: Final[int] = _get_env_int("EPOCHS", 30)
BATCH_SIZE: Final[int] = _get_env_int("BATCH_SIZE", 32)
TRAINING_VERBOSITY: Final[int] = _get_env_int("TRAINING_VERBOSITY", 1)

NLTK_PACKAGES: Final[Tuple[str, ...]] = (
    "punkt_tab",
    "punkt",
    "wordnet",
    "stopwords",
)
STOPWORDS_LANGUAGE: Final[str] = "english"

SENTIMENT_LABELS: Final[Dict[int, str]] = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

DEFAULT_RESPONSES: Final[Dict[str, List[str]]] = {
    "Positive": [
        "Great!",
        "Amazing!",
        "That's great to hear!",
        "Awesome!",
        "I'm happy to hear that!",
        "Wonderful!",
    ],
    "Neutral": [
        "I see. What else can I help with?",
        "Okay.",
        "Got it.",
        "Understood.",
        "Alright.",
        "Thanks for sharing.",
        "Alright, let me know if I can assist further.",
    ],
    "Negative": [
        "I'm sorry to hear that.",
        "That's unfortunate.",
        "I understand your frustration.",
        "I'm sorry.",
        "Let me help you with that.",
    ],
}
FALLBACK_RESPONSE: Final[str] = (
    "I'm here to help. Could you tell me a little more about that?"
)

APP_TITLE: Final[str] = _get_env(
    "APP_TITLE", "Sentiment Analysis Chatbot \U0001F60A"
)
APP_SUBTITLE: Final[str] = _get_env(
    "APP_SUBTITLE",
    "Chat with me and I'll predict the sentiment of your text!",
)
USER_AVATAR: Final[str] = "\U0001F468\u200D\U0001F9B0"
BOT_AVATAR: Final[str] = "\U0001F916"
EMPTY_INPUT_MESSAGE: Final[str] = "Kindly enter a message."

LOG_LEVEL: Final[str] = _get_env("LOG_LEVEL", "INFO").upper()
LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
LOG_DATEFMT: Final[str] = "%Y-%m-%d %H:%M:%S"
LOG_FILE: Final[Path] = LOGS_DIR / "application.log"


def ensure_runtime_directories() -> None:
    """Create runtime directories that must exist before the app starts.

    Datasets and model artefacts are expected to be present already (they are
    shipped with the repository), but the logs directory is created lazily so
    a fresh checkout works without manual setup.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


ensure_runtime_directories()
