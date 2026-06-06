"""Safe IO helpers for model and pickle artefacts.

These helpers consolidate the (otherwise scattered) file IO into one place
with proper logging, descriptive errors and consistent path handling. Every
operation accepts either a :class:`str` or :class:`pathlib.Path`.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Union

from utils.logger import get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]


class ArtifactLoadError(RuntimeError):
    """Raised when a model or pickle artefact cannot be loaded.

    Wrapping the original exception gives callers a single, semantically
    meaningful failure type to catch without having to enumerate every
    possible underlying IO/deserialisation error.
    """


def _coerce_path(path: PathLike) -> Path:
    """Convert ``path`` to a :class:`pathlib.Path`.

    Args:
        path: Either a string or an existing :class:`pathlib.Path`.

    Returns:
        A normalised :class:`pathlib.Path` instance.
    """
    return path if isinstance(path, Path) else Path(path)


def load_pickle(path: PathLike) -> Any:
    """Load and return the object stored in a pickle file.

    Args:
        path: Filesystem location of the pickle file.

    Returns:
        The deserialised Python object.

    Raises:
        ArtifactLoadError: If the file does not exist or cannot be
            deserialised.
    """
    resolved = _coerce_path(path)
    logger.debug("Loading pickle artefact from %s", resolved)

    if not resolved.is_file():
        raise ArtifactLoadError(f"Pickle file not found: {resolved}")

    try:
        with resolved.open("rb") as handle:
            payload = pickle.load(handle)
    except (pickle.UnpicklingError, EOFError, OSError) as exc:
        raise ArtifactLoadError(
            f"Failed to load pickle artefact from {resolved}: {exc}"
        ) from exc

    logger.info("Pickle artefact loaded: %s", resolved.name)
    return payload


def save_pickle(payload: Any, path: PathLike) -> Path:
    """Serialise ``payload`` to ``path`` using the highest pickle protocol.

    The destination directory is created if missing so callers do not need
    to think about filesystem preparation.

    Args:
        payload: Any Python object that supports pickling.
        path: Destination filesystem path for the pickle file.

    Returns:
        The resolved path the artefact was written to.

    Raises:
        ArtifactLoadError: If the artefact cannot be written.
    """
    resolved = _coerce_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    try:
        with resolved.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except (pickle.PicklingError, OSError) as exc:
        raise ArtifactLoadError(
            f"Failed to write pickle artefact to {resolved}: {exc}"
        ) from exc

    logger.info("Pickle artefact saved: %s", resolved)
    return resolved


def load_keras_model(path: PathLike) -> Any:
    """Load a serialised Keras / TensorFlow model from disk.

    Args:
        path: Filesystem location of the ``.h5`` or ``.keras`` model file.

    Returns:
        The deserialised Keras model.

    Raises:
        ArtifactLoadError: If the file is missing or cannot be deserialised.
    """
    resolved = _coerce_path(path)
    logger.debug("Loading Keras model from %s", resolved)

    if not resolved.is_file():
        raise ArtifactLoadError(f"Keras model file not found: {resolved}")

    try:
        from tensorflow.keras.models import load_model
    except ImportError as exc:
        raise ArtifactLoadError(
            "TensorFlow is required to load a Keras model but is not installed."
        ) from exc

    try:
        model = load_model(str(resolved))
    except (OSError, ValueError) as exc:
        raise ArtifactLoadError(
            f"Failed to load Keras model from {resolved}: {exc}"
        ) from exc

    logger.info("Keras model loaded: %s", resolved.name)
    return model
