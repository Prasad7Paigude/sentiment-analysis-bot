"""Configuration package for the Sentiment Analysis Chatbot.

Exposes the centralised :mod:`settings` module so callers can simply do::

    from config import settings

without having to know the internal layout of the package.
"""

from config import settings

__all__ = ["settings"]
