"""Sentiment inference engine.

The :class:`SentimentPredictor` loads the persisted Keras model + TF-IDF
vectorizer once and exposes a tiny, type-safe API to the Streamlit UI
(or any future REST front-end).

The default inference path intentionally mirrors the original ``app.py``
behaviour - the raw user input is passed straight to the vectorizer without
any extra cleaning - so existing model weights produce identical predictions.
The optional ``preprocess`` flag enables the notebook-style cleaning pipeline
for callers that want it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from config import settings
from utils.artifact_io import (
    ArtifactLoadError,
    load_keras_model,
    load_pickle,
)
from utils.logger import get_logger
from utils.text_preprocessing import TextPreprocessor

logger = get_logger(__name__)

PathLike = Union[str, Path]


class SentimentPredictor:
    """High-level wrapper around the trained sentiment classifier.

    Heavy artefacts are loaded eagerly during construction so the first
    prediction request does not incur a multi-second cold-start penalty.
    """

    def __init__(
        self,
        model_path: PathLike = settings.SENTIMENT_MODEL_PATH,
        vectorizer_path: PathLike = settings.TFIDF_VECTORIZER_PATH,
        preprocessor: Optional[TextPreprocessor] = None,
    ) -> None:
        """Eagerly load the model and vectorizer from disk.

        Args:
            model_path: Path to the serialised Keras model.
            vectorizer_path: Path to the pickled TF-IDF vectorizer.
            preprocessor: Optional :class:`TextPreprocessor` instance reused
                when ``preprocess=True`` is passed to :meth:`predict`. A new
                lazy instance is created when omitted.

        Raises:
            ArtifactLoadError: If either artefact cannot be loaded.
        """
        logger.info("Initialising SentimentPredictor.")
        self._model = load_keras_model(model_path)
        self._vectorizer = load_pickle(vectorizer_path)
        self._preprocessor: TextPreprocessor = preprocessor or TextPreprocessor()
        logger.info("SentimentPredictor ready.")

    @property
    def model(self) -> object:
        """Return the underlying Keras model (read-only access)."""
        return self._model

    @property
    def vectorizer(self) -> object:
        """Return the underlying TF-IDF vectorizer (read-only access)."""
        return self._vectorizer

    def _vectorize(self, text: str) -> np.ndarray:
        """Transform a single text input into a dense feature vector.

        Args:
            text: Input string.

        Returns:
            Dense ``(1, n_features)`` numpy array.
        """
        sparse_features = self._vectorizer.transform([text])
        return sparse_features.toarray()

    def predict_class(self, text: str, *, preprocess: bool = False) -> int:
        """Predict the raw integer sentiment class for ``text``.

        Args:
            text: User-supplied input string.
            preprocess: When ``True``, apply the project's standard cleaning
                pipeline before vectorisation. Defaults to ``False`` to remain
                100% compatible with the original ``app.py`` behaviour.

        Returns:
            An integer in ``range(NUM_CLASSES)`` (``0`` = Negative,
            ``1`` = Neutral, ``2`` = Positive).

        Raises:
            ValueError: If ``text`` is ``None`` or empty after stripping.
        """
        if text is None or not str(text).strip():
            raise ValueError("Input text must be a non-empty string.")

        processed = self._preprocessor.clean(text) if preprocess else str(text)
        features = self._vectorize(processed)
        try:
            probabilities = self._model.predict(features, verbose=0)
        except Exception as exc:
            logger.exception("Model inference failed: %s", exc)
            raise

        predicted_class = int(np.argmax(probabilities, axis=1)[0])
        logger.debug(
            "Inference complete: text_len=%d, class=%d", len(processed), predicted_class
        )
        return predicted_class

    def predict_label(self, text: str, *, preprocess: bool = False) -> str:
        """Predict the human-readable sentiment label for ``text``.

        Args:
            text: User-supplied input string.
            preprocess: See :meth:`predict_class`.

        Returns:
            One of ``"Negative"``, ``"Neutral"`` or ``"Positive"`` (or any
            other label declared in :data:`config.settings.SENTIMENT_LABELS`).
        """
        class_index = self.predict_class(text, preprocess=preprocess)
        return settings.SENTIMENT_LABELS.get(class_index, "Unknown")


def load_sentiment_predictor(
    model_path: PathLike = settings.SENTIMENT_MODEL_PATH,
    vectorizer_path: PathLike = settings.TFIDF_VECTORIZER_PATH,
) -> SentimentPredictor:
    """Factory function returning a fully initialised predictor.

    Centralises the construction logic so frameworks like Streamlit can pass
    the function to their caching decorators (``@st.cache_resource``).

    Args:
        model_path: Path to the serialised Keras model.
        vectorizer_path: Path to the pickled TF-IDF vectorizer.

    Returns:
        A ready-to-use :class:`SentimentPredictor`.

    Raises:
        ArtifactLoadError: If any required artefact is missing or corrupt.
    """
    try:
        return SentimentPredictor(
            model_path=model_path,
            vectorizer_path=vectorizer_path,
        )
    except ArtifactLoadError:
        logger.exception("Could not initialise SentimentPredictor.")
        raise
