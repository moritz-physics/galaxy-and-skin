"""Image augmentation transforms for Galaxy10 DECaLS images.

Every transform is a pure function that takes a uint8 ``(256, 256, 3)`` image
and returns a uint8 ``(256, 256, 3)`` image. Transforms that use randomness
take an explicit :class:`numpy.random.Generator` (not a seed) so that
randomness stays explicit and testable.

The training-time transform :func:`augment_train` is restricted to mild,
label-preserving operations (rotation and flips). The remaining transforms
(blur, noise, occlusion) are intended for the robustness evaluation in a later
task and are deliberately destructive.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, rotate

_H, _W, _C = 256, 256, 3


def random_rotation(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rotate the image by a random angle in ``[0, 360)`` degrees.

    Border pixels exposed by the rotation are filled with ``0`` (black),
    simulating empty sky background.

    Parameters
    ----------
    image : numpy.ndarray
        uint8 RGB image of shape ``(256, 256, 3)``.
    rng : numpy.random.Generator
        Random generator used to draw the rotation angle.

    Returns
    -------
    numpy.ndarray
        Rotated uint8 image of shape ``(256, 256, 3)``.
    """
    angle = float(rng.uniform(0.0, 360.0))
    rotated = rotate(
        image.astype(np.float64),
        angle,
        axes=(0, 1),
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
    )
    return np.clip(rotated, 0.0, 255.0).astype(np.uint8)


def random_flip(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a random horizontal and/or vertical flip.

    Parameters
    ----------
    image : numpy.ndarray
        uint8 RGB image of shape ``(256, 256, 3)``.
    rng : numpy.random.Generator
        Random generator used to decide which flips to apply.

    Returns
    -------
    numpy.ndarray
        Flipped uint8 image of shape ``(256, 256, 3)``.
    """
    out = image
    if rng.random() < 0.5:
        out = np.fliplr(out)
    if rng.random() < 0.5:
        out = np.flipud(out)
    return np.ascontiguousarray(out, dtype=np.uint8)


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Blur the image with a Gaussian kernel (simulates PSF degradation).

    Parameters
    ----------
    image : numpy.ndarray
        uint8 RGB image of shape ``(256, 256, 3)``.
    sigma : float
        Standard deviation of the Gaussian kernel, in pixels. Applied to the
        two spatial axes only.

    Returns
    -------
    numpy.ndarray
        Blurred uint8 image of shape ``(256, 256, 3)``.
    """
    blurred = gaussian_filter(image.astype(np.float64), sigma=(sigma, sigma, 0.0))
    return np.clip(blurred, 0.0, 255.0).astype(np.uint8)


def additive_noise(
    image: np.ndarray, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Add zero-mean Gaussian noise and clip back to the valid range.

    Parameters
    ----------
    image : numpy.ndarray
        uint8 RGB image of shape ``(256, 256, 3)``.
    sigma : float
        Standard deviation of the noise on the ``0-255`` intensity scale.
    rng : numpy.random.Generator
        Random generator used to draw the noise.

    Returns
    -------
    numpy.ndarray
        Noisy uint8 image of shape ``(256, 256, 3)``.
    """
    noise = rng.normal(0.0, sigma, size=image.shape)
    noisy = image.astype(np.float64) + noise
    return np.clip(noisy, 0.0, 255.0).astype(np.uint8)


def random_occlusion(
    image: np.ndarray, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Place a black square at a random location (simulates a masked region).

    Parameters
    ----------
    image : numpy.ndarray
        uint8 RGB image of shape ``(256, 256, 3)``.
    size : int
        Side length of the black square, in pixels.
    rng : numpy.random.Generator
        Random generator used to draw the square's position.

    Returns
    -------
    numpy.ndarray
        uint8 image of shape ``(256, 256, 3)`` with the occluded region.
    """
    out = image.copy()
    size = int(min(size, _H, _W))
    y0 = int(rng.integers(0, _H - size + 1))
    x0 = int(rng.integers(0, _W - size + 1))
    out[y0 : y0 + size, x0 : x0 + size, :] = 0
    return out


def augment_train(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply mild, label-preserving training-time augmentation.

    Composes :func:`random_rotation` followed by :func:`random_flip`.

    Parameters
    ----------
    image : numpy.ndarray
        uint8 RGB image of shape ``(256, 256, 3)``.
    rng : numpy.random.Generator
        Random generator shared across the composed transforms.

    Returns
    -------
    numpy.ndarray
        Augmented uint8 image of shape ``(256, 256, 3)``.
    """
    return random_flip(random_rotation(image, rng), rng)
