from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from pyts.datasets import load_gunpoint
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical


def _encode_labels(y_train, y_test):
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)
    num_classes = len(np.unique(y_train_enc))
    return (
        to_categorical(y_train_enc, num_classes),
        to_categorical(y_test_enc, num_classes),
        num_classes,
        encoder,
    )


def _standardize_by_train(X_train: np.ndarray, X_test: np.ndarray):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled.astype("float32"), X_test_scaled.astype("float32")


def load_ucr_tsv(dataset_name: str, data_dir: str | Path = "data/UCR"):
    """
    Load a UCR-style dataset from TSV files.

    Expected structure:
    data/UCR/<DatasetName>/<DatasetName>_TRAIN.tsv
    data/UCR/<DatasetName>/<DatasetName>_TEST.tsv

    UCR TSV files usually store labels in the first column and time-series values
    in the remaining columns.
    """
    data_dir = Path(data_dir)
    folder = data_dir / dataset_name
    train_path = folder / f"{dataset_name}_TRAIN.tsv"
    test_path = folder / f"{dataset_name}_TEST.tsv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Cannot find UCR TSV files for {dataset_name}. Expected:\n"
            f"{train_path}\n{test_path}\n"
            "For quick testing, use --dataset GunPoint."
        )

    train = np.loadtxt(train_path, delimiter="\t")
    test = np.loadtxt(test_path, delimiter="\t")

    y_train, X_train = train[:, 0], train[:, 1:]
    y_test, X_test = test[:, 0], test[:, 1:]

    X_train, X_test = _standardize_by_train(X_train, X_test)
    y_train, y_test, num_classes, encoder = _encode_labels(y_train, y_test)

    return X_train, X_test, y_train, y_test, num_classes, encoder


def load_dataset(dataset_name: str = "GunPoint", data_dir: str | Path = "data/UCR"):
    """Load built-in GunPoint or a local UCR TSV dataset."""
    if dataset_name.lower() == "gunpoint":
        X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)
        X_train, X_test = _standardize_by_train(X_train, X_test)
        y_train, y_test, num_classes, encoder = _encode_labels(y_train, y_test)
        return X_train, X_test, y_train, y_test, num_classes, encoder

    return load_ucr_tsv(dataset_name, data_dir=data_dir)
