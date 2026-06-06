"""Chatbot response selection.

The original ``app.py`` chose responses with ``numpy.random.choice``. The
:class:`ChatbotResponder` keeps that behaviour but wraps it in a class so we
can inject the response bank from disk, fall back gracefully when an unknown
sentiment label arrives, and unit-test the selection logic without spinning
up TensorFlow.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from config import settings
from utils.artifact_io import ArtifactLoadError, load_pickle
from utils.logger import get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]


class ChatbotResponder:
    """Select a chatbot reply based on the predicted sentiment label.

    Attributes:
        responses: Mapping from sentiment label (``"Negative"``, ``"Neutral"``
            or ``"Positive"``) to the list of candidate reply strings.
    """

    def __init__(
        self,
        responses: Optional[Mapping[str, List[str]]] = None,
        fallback: str = settings.FALLBACK_RESPONSE,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Construct a responder from an in-memory mapping.

        Args:
            responses: Optional override for the response bank. Defaults to
                :data:`config.settings.DEFAULT_RESPONSES`.
            fallback: Reply returned when the predicted sentiment label has
                no entry in the response bank.
            rng: Optional :class:`numpy.random.Generator` used for sampling.
                Injecting one makes tests deterministic.
        """
        self._responses: Dict[str, List[str]] = {
            label: list(replies)
            for label, replies in (responses or settings.DEFAULT_RESPONSES).items()
        }
        self._fallback: str = fallback
        self._rng: np.random.Generator = rng or np.random.default_rng()

    @classmethod
    def from_pickle(
        cls,
        path: PathLike = settings.RESPONSES_PATH,
        fallback: str = settings.FALLBACK_RESPONSE,
    ) -> "ChatbotResponder":
        """Build a responder by deserialising the response bank from disk.

        Args:
            path: Filesystem path to the pickled response bank.
            fallback: Reply returned when the predicted sentiment label has
                no entry in the response bank.

        Returns:
            A :class:`ChatbotResponder` initialised with the on-disk bank.

        Raises:
            ArtifactLoadError: If the pickle file is missing, corrupt or has
                an unexpected schema.
        """
        try:
            payload = load_pickle(path)
        except ArtifactLoadError:
            logger.exception("Failed to load chatbot responses from %s", path)
            raise

        if not isinstance(payload, Mapping):
            raise ArtifactLoadError(
                f"Unexpected response bank schema in {path}: expected a mapping, "
                f"got {type(payload).__name__}."
            )

        return cls(responses=payload, fallback=fallback)

    @property
    def responses(self) -> Mapping[str, List[str]]:
        """Return the response bank as an immutable view."""
        return self._responses

    def respond_to_label(self, label: str) -> str:
        """Pick a reply for the supplied sentiment label.

        Args:
            label: Human-readable sentiment label such as ``"Positive"``.

        Returns:
            A randomly chosen reply string. The configured fallback message is
            returned when ``label`` is not in the response bank.
        """
        replies = self._responses.get(label)
        if not replies:
            logger.warning("No replies registered for label '%s'; using fallback.", label)
            return self._fallback
        index = int(self._rng.integers(0, len(replies)))
        return replies[index]

    def respond_to_class(self, class_index: int) -> str:
        """Pick a reply for the raw integer class produced by the model.

        Args:
            class_index: Integer class index returned by
                :meth:`src.model_inference.SentimentPredictor.predict_class`.

        Returns:
            A randomly chosen reply string mapped via
            :data:`config.settings.SENTIMENT_LABELS`.
        """
        label = settings.SENTIMENT_LABELS.get(class_index, "")
        return self.respond_to_label(label)
