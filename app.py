"""Streamlit entry point for the Sentiment Analysis Chatbot.

This module is intentionally thin: every non-presentation concern lives in
``src/`` so the UI can be swapped (FastAPI, gRPC, Gradio, ...) without
touching the inference, response or configuration logic.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from config import settings
from src.model_inference import SentimentPredictor, load_sentiment_predictor
from src.response_generator import ChatbotResponder
from utils.artifact_io import ArtifactLoadError
from utils.logger import get_logger

logger = get_logger(__name__)


@st.cache_resource(show_spinner="Loading the sentiment model...")
def _get_predictor() -> SentimentPredictor:
    """Load the predictor exactly once per Streamlit session.

    Returns:
        A cached :class:`SentimentPredictor` instance.

    Raises:
        ArtifactLoadError: If the underlying model or vectorizer artefacts
            cannot be loaded.
    """
    return load_sentiment_predictor()


@st.cache_resource(show_spinner=False)
def _get_responder() -> ChatbotResponder:
    """Load the chatbot response bank exactly once per Streamlit session.

    The default response bank shipped in :mod:`config.settings` is used as a
    fallback when the pickle file is missing, so first-run users see a
    friendly chatbot rather than a stack trace.

    Returns:
        A cached :class:`ChatbotResponder` instance.
    """
    try:
        return ChatbotResponder.from_pickle()
    except ArtifactLoadError as exc:
        logger.warning(
            "Response bank pickle unavailable (%s); falling back to defaults.", exc
        )
        return ChatbotResponder()


def _render_conversation(user_message: str, bot_message: str) -> None:
    """Render the user / bot exchange in the Streamlit canvas.

    Args:
        user_message: Raw text submitted by the user.
        bot_message: Reply selected by the chatbot.
    """
    st.markdown(f"{settings.USER_AVATAR} **YOU**: {user_message}")
    st.markdown(f"{settings.BOT_AVATAR} **BOT**: {bot_message}")


def main() -> None:
    """Render the Streamlit page and wire up the inference + response flow."""
    st.title(settings.APP_TITLE)
    st.subheader(settings.APP_SUBTITLE)

    try:
        predictor = _get_predictor()
        responder = _get_responder()
    except ArtifactLoadError as exc:
        logger.exception("Fatal startup failure: %s", exc)
        st.error(
            "The sentiment model could not be loaded. "
            "Please verify the files in the `models/` directory and try again."
        )
        return

    user_message = st.text_area("Enter your message here:")

    if not st.button("Send"):
        return

    if not user_message or not user_message.strip():
        st.error(settings.EMPTY_INPUT_MESSAGE)
        return

    try:
        sentiment_class = predictor.predict_class(user_message)
        response = responder.respond_to_class(sentiment_class)
    except ValueError as exc:
        logger.warning("Invalid user input: %s", exc)
        st.error(settings.EMPTY_INPUT_MESSAGE)
        return
    except Exception as exc:
        logger.exception("Unexpected inference failure: %s", exc)
        st.error("Something went wrong while analysing your message. Please try again.")
        return

    _render_conversation(user_message, response)


if __name__ == "__main__":
    main()
