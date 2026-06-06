"""Dataset loading, cleaning and feature-extraction pipeline.

This module mirrors the data preparation logic that originally lived in the
``model_training.ipynb`` Jupyter notebook, repackaged as importable,
type-annotated, log-aware Python functions and classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from config import settings
from utils.artifact_io import save_pickle
from utils.logger import get_logger
from utils.text_preprocessing import TextPreprocessor, download_nltk_packages

logger = get_logger(__name__)

PathLike = Union[str, Path]


@dataclass(frozen=True)
class TrainTestSplit:
    """Container for a TF-IDF vectorised train/test split.

    Attributes:
        x_train: Sparse TF-IDF matrix for the training set.
        x_test: Sparse TF-IDF matrix for the test set.
        y_train: Integer class labels for the training set.
        y_test: Integer class labels for the test set.
        vectorizer: The fitted :class:`TfidfVectorizer` used to produce the
            feature matrices. Persisted alongside the trained model so the
            inference path can replicate the exact transformation.
    """

    x_train: csr_matrix
    x_test: csr_matrix
    y_train: pd.Series
    y_test: pd.Series
    vectorizer: TfidfVectorizer


def load_dataset(path: PathLike = settings.RAW_DATASET_PATH) -> pd.DataFrame:
    """Read the raw Reddit dataset from disk.

    Args:
        path: Filesystem location of the CSV file. Defaults to the path defined
            in :mod:`config.settings`.

    Returns:
        A :class:`pandas.DataFrame` containing the raw dataset.

    Raises:
        FileNotFoundError: If the dataset file is missing.
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Dataset file not found at: {resolved}")

    logger.info("Loading dataset from %s", resolved)
    dataframe = pd.read_csv(resolved)
    logger.info("Dataset shape: %s", dataframe.shape)
    return dataframe


def preprocess_dataframe(
    dataframe: pd.DataFrame,
    text_column: str = settings.TEXT_COLUMN,
    label_column: str = settings.LABEL_COLUMN,
) -> pd.DataFrame:
    """Apply text cleaning + tokenisation to a raw dataframe.

    The cleaning + tokenisation steps mirror the original training notebook
    one-for-one to preserve numerical parity of any re-trained model.

    Args:
        dataframe: Raw dataframe loaded by :func:`load_dataset`.
        text_column: Name of the column containing the raw text.
        label_column: Name of the column containing the sentiment label.

    Returns:
        A new dataframe with two extra columns:

        * ``cleaned_text`` - the result of :meth:`TextPreprocessor.clean`.
        * ``tokens`` - the tokenised representation of ``cleaned_text``.
    """
    download_nltk_packages()
    preprocessor = TextPreprocessor()

    working = dataframe.copy()
    null_text = int(working[text_column].isnull().sum())
    null_label = int(working[label_column].isnull().sum())
    logger.info(
        "Null counts before cleaning: %s=%d, %s=%d.",
        text_column,
        null_text,
        label_column,
        null_label,
    )

    working[settings.CLEANED_TEXT_COLUMN] = working[text_column].apply(preprocessor.clean)
    working[settings.TOKENS_COLUMN] = working[settings.CLEANED_TEXT_COLUMN].apply(
        preprocessor.tokenize
    )
    logger.info("Preprocessed dataframe shape: %s", working.shape)
    return working


def build_feature_matrix(
    cleaned_text: pd.Series,
    max_features: int = settings.TFIDF_MAX_FEATURES,
) -> Tuple[csr_matrix, TfidfVectorizer]:
    """Fit a TF-IDF vectorizer and transform the cleaned text.

    Args:
        cleaned_text: Series of cleaned strings (one document per row).
        max_features: Vocabulary cap forwarded to :class:`TfidfVectorizer`.

    Returns:
        A tuple ``(features, vectorizer)`` where ``features`` is a sparse
        TF-IDF matrix and ``vectorizer`` is the fitted transformer.
    """
    logger.info("Fitting TF-IDF vectorizer with max_features=%d", max_features)
    vectorizer = TfidfVectorizer(max_features=max_features)
    features = vectorizer.fit_transform(cleaned_text)
    logger.info("TF-IDF feature matrix shape: %s", features.shape)
    return features, vectorizer


def split_dataset(
    features: csr_matrix,
    labels: pd.Series,
    test_size: float = settings.TEST_SIZE,
    random_state: int = settings.RANDOM_STATE,
) -> Tuple[csr_matrix, csr_matrix, pd.Series, pd.Series]:
    """Split features and labels into stratification-free train / test sets.

    Args:
        features: TF-IDF feature matrix.
        labels: Aligned label series.
        test_size: Fraction of the data reserved for the test split.
        random_state: Seed for reproducibility.

    Returns:
        ``(x_train, x_test, y_train, y_test)``.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
    )
    logger.info(
        "Train shape: %s | Test shape: %s",
        x_train.shape,
        x_test.shape,
    )
    return x_train, x_test, y_train, y_test


def prepare_training_data(
    dataset_path: PathLike = settings.RAW_DATASET_PATH,
    vectorizer_output_path: PathLike = settings.TFIDF_VECTORIZER_PATH,
) -> TrainTestSplit:
    """End-to-end ingestion pipeline producing model-ready arrays.

    The pipeline performs:

    1. CSV ingestion via :func:`load_dataset`.
    2. Text cleaning + tokenisation via :func:`preprocess_dataframe`.
    3. TF-IDF feature extraction via :func:`build_feature_matrix`.
    4. Train/test split via :func:`split_dataset`.
    5. Persistence of the fitted vectorizer so the inference path can reuse it.

    Args:
        dataset_path: Source CSV file path.
        vectorizer_output_path: Destination for the pickled TF-IDF vectorizer.

    Returns:
        A populated :class:`TrainTestSplit` ready to feed into the trainer.
    """
    raw = load_dataset(dataset_path)
    processed = preprocess_dataframe(raw)
    features, vectorizer = build_feature_matrix(processed[settings.CLEANED_TEXT_COLUMN])
    labels = processed[settings.LABEL_COLUMN]
    x_train, x_test, y_train, y_test = split_dataset(features, labels)

    save_pickle(vectorizer, vectorizer_output_path)

    return TrainTestSplit(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        vectorizer=vectorizer,
    )


def to_dense(matrix: csr_matrix) -> np.ndarray:
    """Convert a sparse matrix to a dense ``numpy`` array.

    Wrapped in a helper to keep call sites symmetric between training and
    inference paths.

    Args:
        matrix: Sparse matrix produced by ``TfidfVectorizer.transform``.

    Returns:
        A dense :class:`numpy.ndarray` representation of ``matrix``.
    """
    return matrix.toarray()
