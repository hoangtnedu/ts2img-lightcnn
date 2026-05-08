from __future__ import annotations

from pathlib import Path
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical

from pyts.datasets import load_gunpoint, fetch_ucr_dataset

try:
    from aeon.datasets import load_classification
except ImportError:
    load_classification = None


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
    X_train = np.asarray(X_train, dtype="float32")
    X_test = np.asarray(X_test, dtype="float32")

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled.astype("float32"), X_test_scaled.astype("float32")


def _to_2d_numpy(X):
    """
    Convert aeon dataset format to 2D numpy array:
    shape = (n_samples, n_timestamps)
    """

    X = np.asarray(X)

    # aeon usually returns: (n_samples, n_channels, n_timestamps)
    if X.ndim == 3:
        if X.shape[1] == 1:
            X = X[:, 0, :]
        elif X.shape[2] == 1:
            X = X[:, :, 0]
        else:
            raise ValueError(
                f"Dataset appears to be multivariate with shape {X.shape}. "
                "This experiment currently supports univariate datasets only."
            )

    if X.ndim != 2:
        raise ValueError(
            f"Expected 2D array after conversion, but got shape {X.shape}."
        )

    return X.astype("float32")


def _load_local_ucr_tsv(dataset_name: str, data_dir: str | Path = "data/UCR"):
    """
    Load local UCR TSV files.

    Expected structure:
    data/UCR/<DatasetName>/<DatasetName>_TRAIN.tsv
    data/UCR/<DatasetName>/<DatasetName>_TEST.tsv
    """

    data_dir = Path(data_dir)
    folder = data_dir / dataset_name

    train_path = folder / f"{dataset_name}_TRAIN.tsv"
    test_path = folder / f"{dataset_name}_TEST.tsv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Cannot find local UCR TSV files for {dataset_name}.\n"
            f"Expected:\n"
            f"{train_path}\n"
            f"{test_path}\n"
        )

    train = np.loadtxt(train_path, delimiter="\t")
    test = np.loadtxt(test_path, delimiter="\t")

    y_train, X_train = train[:, 0], train[:, 1:]
    y_test, X_test = test[:, 0], test[:, 1:]

    X_train, X_test = _standardize_by_train(X_train, X_test)
    y_train, y_test, num_classes, encoder = _encode_labels(y_train, y_test)

    return X_train, X_test, y_train, y_test, num_classes, encoder


def _load_from_aeon(dataset_name: str, data_dir: str | Path = "data/UCR"):
    """
    Load UCR/UEA dataset using aeon.

    Suitable for:
    ECG200, Coffee, FordA, Wafer, etc.
    """

    if load_classification is None:
        raise ImportError("aeon is not installed. Please run: pip install aeon")

    data_dir = Path(data_dir)

    X_train, y_train = load_classification(
        dataset_name,
        split="train",
        extract_path=str(data_dir),
    )

    X_test, y_test = load_classification(
        dataset_name,
        split="test",
        extract_path=str(data_dir),
    )

    X_train = _to_2d_numpy(X_train)
    X_test = _to_2d_numpy(X_test)

    X_train, X_test = _standardize_by_train(X_train, X_test)
    y_train, y_test, num_classes, encoder = _encode_labels(y_train, y_test)

    return X_train, X_test, y_train, y_test, num_classes, encoder


def _load_from_pyts_ucr(dataset_name: str, data_dir: str | Path = "data/UCR"):
    """
    Fallback loader using pyts.
    """

    X_train, X_test, y_train, y_test = fetch_ucr_dataset(
        dataset=dataset_name,
        use_cache=True,
        data_home=str(data_dir),
        return_X_y=True,
    )

    X_train, X_test = _standardize_by_train(X_train, X_test)
    y_train, y_test, num_classes, encoder = _encode_labels(y_train, y_test)

    return X_train, X_test, y_train, y_test, num_classes, encoder


def load_dataset(dataset_name: str = "GunPoint", data_dir: str | Path = "data/UCR"):
    """
    Load dataset for time-series classification.

    Priority:
    1. Local UCR TSV files.
    2. Built-in GunPoint loader from pyts.
    3. aeon loader.
    4. pyts fallback.
    """

    dataset_name = dataset_name.strip()
    data_dir = Path(data_dir)

    local_train_path = data_dir / dataset_name / f"{dataset_name}_TRAIN.tsv"
    local_test_path = data_dir / dataset_name / f"{dataset_name}_TEST.tsv"

    if local_train_path.exists() and local_test_path.exists():
        print(f"Loading local UCR TSV dataset: {dataset_name}")
        return _load_local_ucr_tsv(dataset_name, data_dir=data_dir)

    if dataset_name.lower() == "gunpoint":
        print("Loading built-in GunPoint dataset from pyts.")

        X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)

        X_train, X_test = _standardize_by_train(X_train, X_test)
        y_train, y_test, num_classes, encoder = _encode_labels(y_train, y_test)

        return X_train, X_test, y_train, y_test, num_classes, encoder

    aeon_error = None
    pyts_error = None

    try:
        print(f"Loading dataset with aeon: {dataset_name}")
        return _load_from_aeon(dataset_name, data_dir=data_dir)
    except Exception as error:
        aeon_error = error
        print(f"aeon loader failed for {dataset_name}: {error}")

    try:
        print(f"Trying pyts fallback for dataset: {dataset_name}")
        return _load_from_pyts_ucr(dataset_name, data_dir=data_dir)
    except Exception as error:
        pyts_error = error

    raise RuntimeError(
        f"Cannot load dataset '{dataset_name}'.\n\n"
        f"Please check the dataset name or place TSV files in:\n"
        f"{data_dir}/{dataset_name}/{dataset_name}_TRAIN.tsv\n"
        f"{data_dir}/{dataset_name}/{dataset_name}_TEST.tsv\n\n"
        f"aeon error: {aeon_error}\n"
        f"pyts error: {pyts_error}"
    )