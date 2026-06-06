"""Text preprocessing utilities shared by training and inference code paths.

The :class:`TextPreprocessor` encapsulates the exact cleaning and tokenisation
pipeline used during model training so the inference path is guaranteed to
apply identical preprocessing - a critical invariant for accurate predictions.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Set

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_URL_PATTERN = re.compile(r"http\S+")
_NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z\s]")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def download_nltk_packages(packages: Optional[Iterable[str]] = None) -> None:
    """Download every NLTK corpus required by the project.

    Args:
        packages: Optional iterable of NLTK package identifiers. Defaults to
            :data:`config.settings.NLTK_PACKAGES`.
    """
    try:
        import nltk
    except ImportError:
        logger.error("NLTK is not installed; cannot download corpora.")
        return

    package_list = list(packages) if packages else list(settings.NLTK_PACKAGES)
    for package in package_list:
        try:
            nltk.download(package, quiet=True)
        except Exception as exc:
            logger.warning("Failed to download NLTK package '%s': %s", package, exc)


class TextPreprocessor:
    """Stateless cleaning + tokenisation pipeline for the sentiment model.

    The cleaning steps mirror the training notebook exactly:

    1. Strip URLs.
    2. Remove non-alphabetical characters.
    3. Collapse repeated whitespace.
    4. Remove English stop words.
    5. Lowercase the resulting string.

    Stop words are lazily fetched the first time they are required so importing
    this module never triggers NLTK side-effects.
    """

    def __init__(self, language: str = settings.STOPWORDS_LANGUAGE) -> None:
        """Construct a preprocessor for the requested stop-word language.

        Args:
            language: Stop-word language passed through to
                :func:`nltk.corpus.stopwords.words`.
        """
        self._language: str = language
        self._stop_words: Optional[Set[str]] = None

    @property
    def stop_words(self) -> Set[str]:
        """Return the cached stop-word set, loading it on first access.

        Returns:
            A set of stop words for the configured language.
        """
        if self._stop_words is None:
            self._stop_words = self._load_stop_words()
        return self._stop_words

    def _load_stop_words(self) -> Set[str]:
        """Load the stop-word set from NLTK, downloading corpora if required.

        Returns:
            The set of stop words. An empty set is returned (with a warning
            log line) if NLTK resources are unavailable.
        """
        try:
            from nltk.corpus import stopwords

            return set(stopwords.words(self._language))
        except LookupError:
            logger.info("Stop-words corpus missing; downloading required NLTK data.")
            download_nltk_packages()
            try:
                from nltk.corpus import stopwords

                return set(stopwords.words(self._language))
            except Exception as exc:
                logger.warning("Falling back to empty stop-word set: %s", exc)
                return set()
        except ImportError:
            logger.warning("NLTK is not installed; stop-word filtering disabled.")
            return set()

    def clean(self, text: object) -> str:
        """Apply the standard cleaning pipeline to ``text``.

        Args:
            text: Arbitrary value coerced to :class:`str` before processing.

        Returns:
            The cleaned, lowercased, stop-word-free string. May be empty.
        """
        normalised = str(text) if text is not None else ""
        normalised = _URL_PATTERN.sub("", normalised)
        normalised = _NON_ALPHA_PATTERN.sub("", normalised)
        normalised = _WHITESPACE_PATTERN.sub(" ", normalised).strip()

        if self.stop_words:
            tokens = [word for word in normalised.split() if word not in self.stop_words]
            normalised = " ".join(tokens)

        return normalised.lower()

    def tokenize(self, text: str) -> List[str]:
        """Word-tokenise ``text`` using NLTK's punkt tokenizer.

        Args:
            text: A pre-cleaned string ready for tokenisation.

        Returns:
            A list of word-level tokens. Falls back to a whitespace split
            when NLTK resources are unavailable.
        """
        try:
            from nltk.tokenize import word_tokenize

            return word_tokenize(text)
        except LookupError:
            download_nltk_packages()
            try:
                from nltk.tokenize import word_tokenize

                return word_tokenize(text)
            except Exception as exc:
                logger.warning("Falling back to whitespace tokenizer: %s", exc)
                return text.split()
        except ImportError:
            logger.warning("NLTK is not installed; using whitespace tokenizer.")
            return text.split()
