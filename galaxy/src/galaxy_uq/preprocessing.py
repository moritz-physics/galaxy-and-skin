"""Image preprocessing for the CNN model.

Turns the raw Galaxy10 uint8 (256, 256, 3) tensor into the (N, 3, S, S)
float32 tensor in ``[0, 1]`` that PyTorch expects.
"""

from __future__ import annotations

import logging

import numpy as np
from skimage.transform import resize
from tqdm import tqdm

logger = logging.getLogger(__name__)


def prepare_images_for_cnn(
    images: np.ndarray, target_size: int = 64
) -> np.ndarray:
    """Resize, transpose, and rescale Galaxy10 images for the CNN.

    Parameters
    ----------
    images : numpy.ndarray
        Image tensor of shape ``(N, 256, 256, 3)`` (uint8 or float).
    target_size : int, optional
        Output spatial resolution. Defaults to ``64``.

    Returns
    -------
    numpy.ndarray
        Tensor of shape ``(N, 3, target_size, target_size)`` in ``float32``
        with values in ``[0, 1]``.
    """
    # NOTE: the default 64×64 resize is for CNN input only — it produces a
    # new tensor and does not modify the on-disk dataset, which remains at
    # the original 256×256 resolution in ``data/raw/Galaxy10_DECals.h5``.
    # Classical-feature pipelines (HOG, colour histograms, etc.) read from
    # the raw 256×256 images independently of this function.
    if images.ndim != 4 or images.shape[1:] != (256, 256, 3):
        raise ValueError(
            f"expected images of shape (N, 256, 256, 3), got {images.shape}"
        )
    n = images.shape[0]
    out = np.empty((n, target_size, target_size, 3), dtype=np.float32)
    # skimage.resize is C-backed but the Python-level loop is unavoidable; it's
    # the dominant cost only for the first preprocessing pass.
    for i in tqdm(range(n), desc=f"resize→{target_size}", leave=False):
        out[i] = resize(
            images[i],
            (target_size, target_size, 3),
            preserve_range=False,
            anti_aliasing=True,
        ).astype(np.float32)
    # Channels-first layout for PyTorch; .copy() ensures the array is
    # contiguous after the transpose so subsequent torch.from_numpy calls
    # don't suffer stride penalties on the GPU.
    return np.ascontiguousarray(out.transpose(0, 3, 1, 2))
