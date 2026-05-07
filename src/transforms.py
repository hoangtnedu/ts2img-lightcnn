from __future__ import annotations

import numpy as np
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
from scipy.signal import stft
from skimage.transform import resize


def _to_float32_channel(X_img: np.ndarray) -> np.ndarray:
    X_img = X_img.astype("float32")
    if X_img.ndim == 3:
        X_img = X_img[..., np.newaxis]
    return X_img


def to_gaf(X: np.ndarray, image_size: int = 64, method: str = "summation") -> np.ndarray:
    """Gramian Angular Field image representation."""
    transformer = GramianAngularField(image_size=image_size, method=method)
    return _to_float32_channel(transformer.fit_transform(X))


def to_mtf(X: np.ndarray, image_size: int = 64, n_bins: int = 8) -> np.ndarray:
    """Markov Transition Field image representation."""
    transformer = MarkovTransitionField(image_size=image_size, n_bins=n_bins)
    return _to_float32_channel(transformer.fit_transform(X))


def to_rp(X: np.ndarray, image_size: int = 64) -> np.ndarray:
    """Recurrence Plot image representation."""
    transformer = RecurrencePlot()
    X_img = transformer.fit_transform(X)

    resized = []
    for img in X_img:
        resized.append(resize(img, (image_size, image_size), anti_aliasing=True))

    return _to_float32_channel(np.asarray(resized))


def to_stft_image(X: np.ndarray, image_size: int = 64) -> np.ndarray:
    """Time-frequency image using Short-Time Fourier Transform magnitude."""
    images = []
    for x in X:
        _, _, zxx = stft(x)
        img = np.abs(zxx)
        img = resize(img, (image_size, image_size), anti_aliasing=True)
        images.append(img)
    return _to_float32_channel(np.asarray(images))


def transform_1d_to_2d(X: np.ndarray, representation: str, image_size: int = 64) -> np.ndarray:
    representation = representation.lower().strip()

    if representation in {"gaf", "gasf"}:
        return to_gaf(X, image_size=image_size, method="summation")

    if representation == "gadf":
        return to_gaf(X, image_size=image_size, method="difference")

    if representation == "mtf":
        return to_mtf(X, image_size=image_size)

    if representation == "rp":
        return to_rp(X, image_size=image_size)

    if representation == "stft":
        return to_stft_image(X, image_size=image_size)

    raise ValueError(
        f"Unknown representation: {representation}. "
        "Use one of: gaf, gasf, gadf, mtf, rp, stft."
    )
