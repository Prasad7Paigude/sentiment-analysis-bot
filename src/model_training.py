"""Model architecture, training loop and evaluation utilities.

The neural-network topology, loss, optimiser and label-encoding scheme are an
exact 1:1 port of the original ``model_training.ipynb`` notebook. Only the
surrounding scaffolding (logging, type hints, modularity, IO safety) has been
modernised.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tensorflow.keras.utils import to_categorical

from config import settings
from src.data_ingestion import TrainTestSplit, prepare_training_data, to_dense
from utils.logger import get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]


def encode_labels(labels: pd.Series) -> np.ndarray:
    """One-hot encode ``labels`` after shifting them into the ``[0, N)`` range.

    The original dataset uses ``{-1, 0, 1}`` to denote sentiment, but
    ``to_categorical`` expects non-negative integers. The label offset is
    applied here to preserve the exact behaviour of the training notebook.

    Args:
        labels: Series of raw ``{-1, 0, 1}`` sentiment labels.

    Returns:
        A ``(n_samples, num_classes)`` one-hot encoded numpy array.
    """
    return to_categorical(labels + settings.LABEL_OFFSET, num_classes=settings.NUM_CLASSES)


def build_model(input_dim: int) -> tf.keras.Model:
    """Construct the compiled feed-forward sentiment classifier.

    The architecture is identical to the one defined in the original notebook
    so any pre-trained weights remain fully compatible::

        Dense(256, relu) -> Dropout(0.5)
            -> Dense(128, relu) -> Dropout(0.5)
            -> Dense(3, softmax)

    Args:
        input_dim: Size of the TF-IDF feature vector (vocabulary length).

    Returns:
        A compiled :class:`tf.keras.Model` ready for training.
    """
    logger.info("Building sentiment classification model (input_dim=%d).", input_dim)
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Dense(
            settings.DENSE_UNITS_LAYER_1,
            input_shape=(input_dim,),
            activation=settings.HIDDEN_ACTIVATION,
        )
    )
    model.add(tf.keras.layers.Dropout(settings.DROPOUT_RATE))
    model.add(
        tf.keras.layers.Dense(
            settings.DENSE_UNITS_LAYER_2,
            activation=settings.HIDDEN_ACTIVATION,
        )
    )
    model.add(tf.keras.layers.Dropout(settings.DROPOUT_RATE))
    model.add(
        tf.keras.layers.Dense(
            settings.NUM_CLASSES,
            activation=settings.OUTPUT_ACTIVATION,
        )
    )

    model.compile(
        loss=settings.LOSS_FUNCTION,
        optimizer=settings.OPTIMIZER,
        metrics=list(settings.METRICS),
    )
    return model


def train_model(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = settings.EPOCHS,
    batch_size: int = settings.BATCH_SIZE,
    verbose: int = settings.TRAINING_VERBOSITY,
) -> tf.keras.callbacks.History:
    """Fit ``model`` and return the Keras training history.

    Args:
        model: A compiled model produced by :func:`build_model`.
        x_train: Dense training feature matrix.
        y_train: One-hot encoded training labels.
        x_test: Dense validation feature matrix.
        y_test: One-hot encoded validation labels.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        verbose: Verbosity level forwarded to Keras (``0``, ``1`` or ``2``).

    Returns:
        The :class:`tf.keras.callbacks.History` object returned by
        ``model.fit``.
    """
    logger.info(
        "Training model: epochs=%d, batch_size=%d, train_samples=%d.",
        epochs,
        batch_size,
        x_train.shape[0],
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        verbose=verbose,
    )
    logger.info("Training completed.")
    return history


def save_model(model: tf.keras.Model, path: PathLike = settings.SENTIMENT_MODEL_PATH) -> Path:
    """Persist a trained Keras model to disk.

    Args:
        model: The trained Keras model to serialise.
        path: Destination filesystem path. The legacy HDF5 format is preserved
            (``.h5``) to remain compatible with the existing application code.

    Returns:
        The resolved path that was written to.

    Raises:
        OSError: If the model cannot be written to ``path``.
    """
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        model.save(str(resolved))
    except OSError as exc:
        logger.error("Failed to save model to %s: %s", resolved, exc)
        raise
    logger.info("Model saved to %s", resolved)
    return resolved


def evaluate_model(
    model: tf.keras.Model,
    x_test: np.ndarray,
    y_test_encoded: np.ndarray,
    y_test_raw: pd.Series,
) -> Dict[str, Any]:
    """Evaluate a trained model and emit a full diagnostic report.

    Args:
        model: The trained Keras model.
        x_test: Dense test feature matrix.
        y_test_encoded: One-hot encoded ground truth labels.
        y_test_raw: Original ``{-1, 0, 1}`` integer labels (used for the
            classification report and confusion matrix).

    Returns:
        Dictionary with ``loss``, ``accuracy``, ``classification_report`` and
        ``confusion_matrix`` keys.
    """
    loss, accuracy = model.evaluate(x_test, y_test_encoded, verbose=0)
    logger.info("Test accuracy: %.2f%%", accuracy * 100)

    predictions = model.predict(x_test, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1) - settings.LABEL_OFFSET

    report = classification_report(y_test_raw, predicted_labels, output_dict=True)
    matrix = confusion_matrix(y_test_raw, predicted_labels)

    logger.info(
        "Classification report:\n%s",
        classification_report(y_test_raw, predicted_labels),
    )
    logger.info("Confusion matrix:\n%s", matrix)

    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_test_raw, predicted_labels)),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


def run_training_pipeline(
    dataset_path: PathLike = settings.RAW_DATASET_PATH,
    model_output_path: PathLike = settings.SENTIMENT_MODEL_PATH,
    vectorizer_output_path: PathLike = settings.TFIDF_VECTORIZER_PATH,
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """Execute the full ingestion + training + evaluation flow.

    Args:
        dataset_path: Location of the raw CSV dataset.
        model_output_path: Destination for the serialised Keras model.
        vectorizer_output_path: Destination for the fitted TF-IDF vectorizer.

    Returns:
        A tuple ``(model, evaluation_metrics)`` where ``model`` is the trained
        Keras model and ``evaluation_metrics`` is the dictionary returned by
        :func:`evaluate_model`.
    """
    logger.info("Starting end-to-end training pipeline.")
    split: TrainTestSplit = prepare_training_data(
        dataset_path=dataset_path,
        vectorizer_output_path=vectorizer_output_path,
    )

    x_train_dense = to_dense(split.x_train)
    x_test_dense = to_dense(split.x_test)
    y_train_encoded = encode_labels(split.y_train)
    y_test_encoded = encode_labels(split.y_test)

    model = build_model(input_dim=x_train_dense.shape[1])
    train_model(
        model=model,
        x_train=x_train_dense,
        y_train=y_train_encoded,
        x_test=x_test_dense,
        y_test=y_test_encoded,
    )
    save_model(model, model_output_path)
    metrics = evaluate_model(model, x_test_dense, y_test_encoded, split.y_test)

    logger.info("Training pipeline finished successfully.")
    return model, metrics
