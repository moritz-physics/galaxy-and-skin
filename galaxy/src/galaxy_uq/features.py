"""Handcrafted feature extraction for Galaxy10 DECaLS images.

This module turns a single uint8 RGB image into a fixed-length, deterministic
float64 feature vector. Features are organised into six groups:

==================  =====  ====================================================
Group               Count  Contents
==================  =====  ====================================================
photometric            16  mean/std/skew/kurtosis of R, G, B, grayscale
cas                    10  concentration, asymmetry, smoothness, gini, m20,
                           ellipticity, orientation, Sersic-like concentration,
                           half-light radius, Petrosian radius
radial                 20  10-bin grayscale radial profile + 10-bin (G-R) color
                           ratio radial profile
hu                      7  log-transformed Hu invariant moments
hog                    36  HOG descriptor (2x2 cells, 9 orientations)
gabor                  24  mean/std of magnitude for 12 Gabor filters
==================  =====  ====================================================

Total: 113 features (>= 100 as required).

The CAS and Gini-M20 morphology indicators (Conselice 2003; Lotz et al. 2004)
are implemented from scratch in NumPy. Following standard practice, the sky
background is estimated from the image corners and subtracted, and the
statistics are measured within a circular aperture of ``1.5 x`` the Petrosian
radius rather than over the whole frame. All denominators are guarded:
degenerate inputs (e.g. a black image, an empty aperture, an unreachable
Petrosian radius, a zero moment denominator) fall back to ``0.0`` rather than
producing ``NaN``/``Inf``.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.feature import hog
from skimage.filters import gabor_kernel
from skimage.measure import (
    moments,
    moments_central,
    moments_hu,
    moments_normalized,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Luminosity weights used for the RGB -> grayscale conversion.
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)

# Number of features contributed by each group, in output order.
GROUP_SIZES: dict[str, int] = {
    "photometric": 16,
    "cas": 10,
    "radial": 20,
    "hu": 7,
    "hog": 36,
    "gabor": 24,
}

# Gabor filter bank parameters.
_GABOR_FREQS: tuple[float, ...] = (0.05, 0.15, 0.25)
_GABOR_THETAS: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)

_EPS = 1e-12


def _build_gabor_fft_kernels() -> tuple[int, list[np.ndarray], list[tuple[int, int]]]:
    """Pre-compute padded FFTs of the Gabor filter bank.

    The 12 (3 frequency x 4 orientation) complex Gabor kernels are built once
    and stored as FFTs on a common padded grid. This lets :func:`_gabor_features`
    convolve via the FFT (O(N log N)) instead of the very slow spatial
    convolution that large low-frequency kernels would otherwise require.

    Returns
    -------
    size : int
        Side length of the square FFT grid.
    kernel_ffts : list of numpy.ndarray
        FFT of each zero-padded complex Gabor kernel.
    kernel_shapes : list of tuple of int
        ``(height, width)`` of each (unpadded) kernel.
    """
    kernels = [
        gabor_kernel(frequency=freq, theta=theta)
        for freq in _GABOR_FREQS
        for theta in _GABOR_THETAS
    ]
    max_k = max(max(k.shape) for k in kernels)
    # Grid must hold the full linear convolution (256 + max_k - 1); 512 is an
    # FFT-friendly size that comfortably exceeds this for the chosen bank.
    size = 512
    if 256 + max_k - 1 > size:
        raise RuntimeError(f"Gabor kernel too large ({max_k}) for grid {size}")
    kernel_ffts: list[np.ndarray] = []
    kernel_shapes: list[tuple[int, int]] = []
    for kernel in kernels:
        padded = np.zeros((size, size), dtype=np.complex128)
        padded[: kernel.shape[0], : kernel.shape[1]] = kernel
        kernel_ffts.append(np.fft.fft2(padded))
        kernel_shapes.append((kernel.shape[0], kernel.shape[1]))
    return size, kernel_ffts, kernel_shapes


_GABOR_FFT_SIZE, _GABOR_KERNEL_FFTS, _GABOR_KERNEL_SHAPES = _build_gabor_fft_kernels()


def _build_feature_names() -> list[str]:
    """Construct the human-readable feature names in output order.

    Returns
    -------
    list of str
        Names aligned one-to-one with the vector from :func:`extract_features`.
    """
    names: list[str] = []
    for channel in ("R", "G", "B", "gray"):
        for stat in ("mean", "std", "skew", "kurtosis"):
            names.append(f"photometric_{channel}_{stat}")
    names += [
        "cas_concentration",
        "cas_asymmetry",
        "cas_smoothness",
        "gini",
        "m20",
        "cas_ellipticity",
        "cas_orientation",
        "sersic_concentration",
        "r_half",
        "petrosian_radius",
    ]
    names += [f"radial_bin_{i}" for i in range(10)]
    names += [f"color_radial_bin_{i}" for i in range(10)]
    names += [f"hu_{i}" for i in range(7)]
    for by in range(2):
        for bx in range(2):
            for orient in range(9):
                names.append(f"hog_{by}_{bx}_orient{orient}")
    for fi in range(len(_GABOR_FREQS)):
        for oi in range(len(_GABOR_THETAS)):
            names.append(f"gabor_freq{fi}_orient{oi}_mean")
            names.append(f"gabor_freq{fi}_orient{oi}_std")
    return names


FEATURE_NAMES: list[str] = _build_feature_names()


# --------------------------------------------------------------------------
# Group 1: photometric statistics
# --------------------------------------------------------------------------
def _moment_stats(channel: np.ndarray) -> tuple[float, float, float, float]:
    """Return mean, std, skewness and excess kurtosis of a flattened array.

    Parameters
    ----------
    channel : numpy.ndarray
        Input array; flattened internally.

    Returns
    -------
    tuple of float
        ``(mean, std, skewness, excess_kurtosis)``. Skewness and kurtosis are
        ``0.0`` when the variance is effectively zero.
    """
    x = channel.astype(np.float64).ravel()
    mu = float(x.mean())
    d = x - mu
    var = float((d**2).mean())
    if var < _EPS:
        return mu, 0.0, 0.0, 0.0
    std = float(np.sqrt(var))
    skew = float((d**3).mean() / var**1.5)
    kurt = float((d**4).mean() / var**2 - 3.0)
    return mu, std, skew, kurt


def _photometric_features(rgb: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Compute the 16 photometric statistics (Group 1).

    Parameters
    ----------
    rgb : numpy.ndarray
        Float image in ``[0, 1]`` of shape ``(256, 256, 3)``.
    gray : numpy.ndarray
        Float grayscale image in ``[0, 1]`` of shape ``(256, 256)``.

    Returns
    -------
    numpy.ndarray
        16-element feature vector.
    """
    out: list[float] = []
    for channel in (rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2], gray):
        out.extend(_moment_stats(channel))
    return np.asarray(out, dtype=np.float64)


# --------------------------------------------------------------------------
# Group 2: CAS + Gini-M20 morphology
# --------------------------------------------------------------------------
def _centroid(gray: np.ndarray) -> tuple[float, float]:
    """Return the intensity-weighted centroid ``(cy, cx)`` of a grayscale image.

    Falls back to the geometric centre when the total flux is negligible.
    """
    total = float(gray.sum())
    h, w = gray.shape
    if total < _EPS:
        return (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    cy = float((gray * yy).sum() / total)
    cx = float((gray * xx).sum() / total)
    return cy, cx


def _gini(values: np.ndarray) -> float:
    """Compute the standard Gini coefficient of a set of pixel values.

    Parameters
    ----------
    values : numpy.ndarray
        Pixel intensities; absolute values are used.

    Returns
    -------
    float
        Gini coefficient, or ``0.0`` if the mean is effectively zero.
    """
    x = np.sort(np.abs(values.astype(np.float64).ravel()))
    n = x.size
    if n < 2:
        return 0.0
    mean = float(x.mean())
    if mean < _EPS:
        return 0.0
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float(np.sum((2.0 * idx - n - 1.0) * x) / (mean * n * (n - 1)))


def _m20(flux: np.ndarray, d2: np.ndarray, total: float) -> float:
    """Compute the log10 M20 statistic (Lotz et al. 2004).

    Parameters
    ----------
    flux : numpy.ndarray
        1-D array of pixel fluxes over the measurement aperture.
    d2 : numpy.ndarray
        1-D array of squared distances from the centroid, aligned with
        ``flux``.
    total : float
        Total flux over the aperture.

    Returns
    -------
    float
        ``log10`` of the second-order moment ratio, or ``0.0`` when the total
        moment is degenerate.
    """
    m_tot = float(np.sum(flux * d2))
    if m_tot < _EPS:
        return 0.0
    order = np.argsort(flux)[::-1]
    f_sorted = flux[order]
    d2_sorted = d2[order]
    cum = np.cumsum(f_sorted)
    k = int(np.searchsorted(cum, 0.2 * total)) + 1
    k = min(k, f_sorted.size)
    m_bright = float(np.sum(f_sorted[:k] * d2_sorted[:k]))
    ratio = m_bright / m_tot
    if ratio < _EPS:
        return 0.0
    return float(np.log10(ratio))


def _ellipticity_orientation(
    w: np.ndarray, ys: np.ndarray, xs: np.ndarray
) -> tuple[float, float]:
    """Compute ellipticity and major-axis orientation from central moments.

    Parameters
    ----------
    w : numpy.ndarray
        1-D intensity weights of the support pixels.
    ys, xs : numpy.ndarray
        1-D pixel coordinates aligned with ``w``.

    Returns
    -------
    tuple of float
        ``(ellipticity, orientation)`` where ellipticity is ``1 - b/a`` and
        orientation is in radians within ``[0, pi]``. Both are ``0.0`` for a
        degenerate support.
    """
    total_w = float(w.sum())
    if w.size < 3 or total_w < _EPS:
        return 0.0, 0.0
    yc = float((w * ys).sum() / total_w)
    xc = float((w * xs).sum() / total_w)
    mu20 = float((w * (xs - xc) ** 2).sum() / total_w)
    mu02 = float((w * (ys - yc) ** 2).sum() / total_w)
    mu11 = float((w * (xs - xc) * (ys - yc)).sum() / total_w)
    common = (mu20 + mu02) / 2.0
    diff = float(np.sqrt(((mu20 - mu02) / 2.0) ** 2 + mu11**2))
    a2 = common + diff
    b2 = max(common - diff, 0.0)
    ellipticity = 0.0 if a2 < _EPS else float(1.0 - np.sqrt(b2 / a2))
    orientation = 0.5 * float(np.arctan2(2.0 * mu11, mu20 - mu02))
    if orientation < 0.0:
        orientation += np.pi
    return ellipticity, orientation


def _petrosian_radius(gray: np.ndarray, r: np.ndarray) -> float:
    """Compute the Petrosian radius using 1-pixel annuli out to 128 pixels.

    The Petrosian radius is the smallest radius at which the surface brightness
    in the local annulus drops to ``0.2`` times the mean surface brightness
    interior to that radius.

    Parameters
    ----------
    gray : numpy.ndarray
        Grayscale image.
    r : numpy.ndarray
        Per-pixel radial distance from the centroid.

    Returns
    -------
    float
        Petrosian radius in pixels; ``128.0`` if the threshold is never met.
    """
    # Bin pixels into 1-pixel-wide integer-radius annuli in a single pass.
    bins = np.clip(np.floor(r).astype(np.int64).ravel(), 0, 128)
    flux = gray.ravel()
    flux_per_bin = np.bincount(bins, weights=flux, minlength=129)
    count_per_bin = np.bincount(bins, minlength=129).astype(np.float64)

    # Annulus surface brightness, and mean surface brightness interior to it.
    sb_annulus = np.divide(
        flux_per_bin, count_per_bin, out=np.zeros(129), where=count_per_bin > 0
    )
    cum_count = np.cumsum(count_per_bin)
    sb_inside = np.divide(
        np.cumsum(flux_per_bin), cum_count, out=np.zeros(129), where=cum_count > 0
    )
    # Bin index ``radius - 1`` holds the annulus [radius-1, radius) and the
    # interior r < radius.
    for radius in range(1, 129):
        i = radius - 1
        if count_per_bin[i] == 0 or sb_inside[i] < _EPS:
            continue
        if sb_annulus[i] / sb_inside[i] <= 0.2:
            return float(radius)
    return 128.0


def _estimate_background(gray: np.ndarray) -> float:
    """Estimate the sky-background level from the image corners.

    Galaxy10 images are galaxy-centred, so the four 32x32 corner patches are
    dominated by sky. Their median is a robust background estimate.

    Parameters
    ----------
    gray : numpy.ndarray
        Float grayscale image in ``[0, 1]``.

    Returns
    -------
    float
        Estimated background level.
    """
    c = 32
    corners = np.concatenate(
        [
            gray[:c, :c].ravel(),
            gray[:c, -c:].ravel(),
            gray[-c:, :c].ravel(),
            gray[-c:, -c:].ravel(),
        ]
    )
    return float(np.median(corners))


def _cas_features(gray: np.ndarray) -> np.ndarray:
    """Compute the 10 CAS + Gini-M20 morphology features (Group 2).

    The sky background is estimated from the image corners and subtracted, and
    every statistic is measured within a circular aperture of ``1.5 x`` the
    Petrosian radius centred on the brightness centroid (Conselice 2003; Lotz
    et al. 2004). This keeps the large sky region of the ``256 x 256`` frame
    from dominating the morphology measurements.

    Parameters
    ----------
    gray : numpy.ndarray
        Float grayscale image in ``[0, 1]`` of shape ``(256, 256)``.

    Returns
    -------
    numpy.ndarray
        10-element feature vector ordered as concentration, asymmetry,
        smoothness, gini, m20, ellipticity, orientation, Sersic-like
        concentration, half-light radius, Petrosian radius. An all-zero vector
        is returned for a degenerate (black / empty-aperture) image.
    """
    h, w = gray.shape
    sky = _estimate_background(gray)
    img = np.clip(gray - sky, 0.0, None)
    if float(img.sum()) < _EPS:
        return np.zeros(10, dtype=np.float64)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cy, cx = _centroid(img)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    petrosian = _petrosian_radius(img, r)

    # Circular measurement aperture at 1.5 x the Petrosian radius.
    aperture = r <= 1.5 * petrosian
    flux_ap = img[aperture]
    total = float(flux_ap.sum())
    if total < _EPS or flux_ap.size < 10:
        return np.zeros(10, dtype=np.float64)
    r_ap = r[aperture]
    yy_ap = yy[aperture]
    xx_ap = xx[aperture]

    # Cumulative flux vs radius within the aperture (for enclosed-flux radii).
    order = np.argsort(r_ap)
    r_sorted = r_ap[order]
    f_sorted = flux_ap[order]
    cum = np.cumsum(f_sorted)

    def _radius_at(frac: float) -> float:
        idx = int(np.searchsorted(cum, frac * total))
        idx = min(idx, r_sorted.size - 1)
        return float(r_sorted[idx])

    r20 = _radius_at(0.2)
    r80 = _radius_at(0.8)
    r_half = _radius_at(0.5)
    concentration = r80 / r20 if r20 > 1e-6 else 0.0

    # Asymmetry: rotate 180 degrees about the centroid, compared within aperture.
    rotated = map_coordinates(
        img, [2.0 * cy - yy, 2.0 * cx - xx], order=1, mode="constant", cval=0.0
    )
    abs_total = float(np.abs(flux_ap).sum())
    asymmetry = (
        float(np.abs(img - rotated)[aperture].sum() / (2.0 * abs_total))
        if abs_total > _EPS
        else 0.0
    )

    # Smoothness (clumpiness): residual after Gaussian smoothing, within aperture.
    sigma = 0.25 * petrosian
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = 5.0
    smoothed = gaussian_filter(img, sigma)
    smoothness = (
        float(np.abs(img - smoothed)[aperture].sum() / abs_total)
        if abs_total > _EPS
        else 0.0
    )

    gini = _gini(flux_ap)
    d2_ap = (yy_ap - cy) ** 2 + (xx_ap - cx) ** 2
    m20 = _m20(flux_ap, d2_ap, total)

    # Ellipticity / orientation from aperture pixels brighter than the
    # aperture mean, weighted by intensity.
    bright = flux_ap > float(flux_ap.mean())
    ellipticity, orientation = _ellipticity_orientation(
        flux_ap[bright], yy_ap[bright], xx_ap[bright]
    )

    # Sersic-like concentration: enclosed flux fraction within 0.3 * R_half.
    inner_mask = r_sorted <= 0.3 * r_half
    sersic = float(f_sorted[inner_mask].sum() / total) if inner_mask.any() else 0.0

    return np.array(
        [
            concentration,
            asymmetry,
            smoothness,
            gini,
            m20,
            ellipticity,
            orientation,
            sersic,
            r_half,
            petrosian,
        ],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------
# Group 3: radial profile
# --------------------------------------------------------------------------
def _radial_features(gray: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Compute the 20 radial-profile features (Group 3).

    Parameters
    ----------
    gray : numpy.ndarray
        Float grayscale image in ``[0, 1]``.
    rgb : numpy.ndarray
        Float RGB image in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        20-element vector: 10 grayscale-profile bins followed by 10 ``(G-R)``
        color-ratio bins, both centred on the brightness centroid and spanning
        radii ``0`` to ``100`` pixels. Empty bins are ``0.0``.
    """
    h, w = gray.shape
    cy, cx = _centroid(gray)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    edges = np.linspace(0.0, 100.0, 11)

    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    color_ratio = (green - red) / (red + green + 1e-8)

    gray_profile = np.zeros(10, dtype=np.float64)
    color_profile = np.zeros(10, dtype=np.float64)
    for i in range(10):
        bin_mask = (r >= edges[i]) & (r < edges[i + 1])
        if bin_mask.any():
            gray_profile[i] = float(gray[bin_mask].mean())
            color_profile[i] = float(color_ratio[bin_mask].mean())
    return np.concatenate([gray_profile, color_profile])


# --------------------------------------------------------------------------
# Group 4: Hu moments
# --------------------------------------------------------------------------
def _hu_features(gray: np.ndarray) -> np.ndarray:
    """Compute the 7 log-transformed Hu invariant moments (Group 4).

    Parameters
    ----------
    gray : numpy.ndarray
        Float grayscale image in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        7-element vector; the transform applied is
        ``-sign(h) * log10(|h| + 1e-12)``. Non-finite or zero raw moments map
        to ``0.0``.
    """
    out = np.zeros(7, dtype=np.float64)
    m = moments(gray)
    if m[0, 0] < _EPS:
        return out
    cy = m[1, 0] / m[0, 0]
    cx = m[0, 1] / m[0, 0]
    mu = moments_central(gray, center=(cy, cx))
    nu = moments_normalized(mu)
    hu = moments_hu(np.nan_to_num(nu, nan=0.0, posinf=0.0, neginf=0.0))
    for i, value in enumerate(hu):
        if not np.isfinite(value) or value == 0.0:
            out[i] = 0.0
        else:
            out[i] = float(-np.sign(value) * np.log10(np.abs(value) + 1e-12))
    return out


# --------------------------------------------------------------------------
# Group 5: HOG texture
# --------------------------------------------------------------------------
def _hog_features(gray: np.ndarray) -> np.ndarray:
    """Compute the 36-element HOG descriptor (Group 5).

    Parameters
    ----------
    gray : numpy.ndarray
        Float grayscale image in ``[0, 1]`` of shape ``(256, 256)``.

    Returns
    -------
    numpy.ndarray
        36-element HOG feature vector (2x2 cells, 9 orientations).
    """
    descriptor = hog(
        gray,
        orientations=9,
        pixels_per_cell=(128, 128),
        cells_per_block=(1, 1),
        feature_vector=True,
    )
    out = np.zeros(36, dtype=np.float64)
    out[: min(36, descriptor.size)] = descriptor[:36]
    return out


# --------------------------------------------------------------------------
# Group 6: Gabor texture
# --------------------------------------------------------------------------
def _gabor_features(gray: np.ndarray) -> np.ndarray:
    """Compute the 24 Gabor texture features (Group 6).

    Each of the 12 complex Gabor filters is applied by FFT convolution against
    the pre-computed kernel bank; the central ``256 x 256`` region of the
    response is kept and its magnitude summarised.

    Parameters
    ----------
    gray : numpy.ndarray
        Float grayscale image in ``[0, 1]`` of shape ``(256, 256)``.

    Returns
    -------
    numpy.ndarray
        24-element vector: mean and std of the magnitude response for each of
        the 12 (3 frequency x 4 orientation) Gabor filters.
    """
    h, w = gray.shape
    padded = np.zeros((_GABOR_FFT_SIZE, _GABOR_FFT_SIZE), dtype=np.float64)
    padded[:h, :w] = gray
    image_fft = np.fft.fft2(padded)

    out: list[float] = []
    for kernel_fft, (kh, kw) in zip(_GABOR_KERNEL_FFTS, _GABOR_KERNEL_SHAPES):
        response = np.fft.ifft2(image_fft * kernel_fft)
        # Crop the central region aligned with the original image extent.
        oy, ox = kh // 2, kw // 2
        magnitude = np.abs(response[oy : oy + h, ox : ox + w])
        out.append(float(magnitude.mean()))
        out.append(float(magnitude.std()))
    return np.asarray(out, dtype=np.float64)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def extract_features(image: np.ndarray) -> np.ndarray:
    """Extract the full handcrafted feature vector from a single image.

    Parameters
    ----------
    image : numpy.ndarray
        A uint8 RGB image of shape ``(256, 256, 3)``.

    Returns
    -------
    numpy.ndarray
        Deterministic 1-D ``float64`` feature vector of length
        ``len(FEATURE_NAMES)`` (113).

    Raises
    ------
    ValueError
        If ``image`` does not have shape ``(256, 256, 3)``.
    """
    if image.shape != (256, 256, 3):
        raise ValueError(
            f"expected image shape (256, 256, 3), got {image.shape}"
        )
    rgb = image.astype(np.float64) / 255.0
    gray = rgb @ _LUMA

    parts = [
        _photometric_features(rgb, gray),
        _cas_features(gray),
        _radial_features(gray, rgb),
        _hu_features(gray),
        _hog_features(gray),
        _gabor_features(gray),
    ]
    return np.concatenate(parts).astype(np.float64)


def extract_all_features(
    images: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract features for every image in a dataset.

    Parameters
    ----------
    images : numpy.ndarray
        Image tensor of shape ``(N, 256, 256, 3)`` and dtype ``uint8``.
    labels : numpy.ndarray
        Integer class labels of shape ``(N,)``.

    Returns
    -------
    feature_matrix : numpy.ndarray
        ``float64`` array of shape ``(N, D)``.
    labels : numpy.ndarray
        Labels as ``int64`` of shape ``(N,)``.
    feature_names : list of str
        The ``D`` feature names, aligned with the columns of the matrix.
    """
    n = images.shape[0]
    d = len(FEATURE_NAMES)
    feature_matrix = np.zeros((n, d), dtype=np.float64)
    for i in tqdm(range(n), desc="extracting features"):
        feature_matrix[i] = extract_features(images[i])
    logger.info("Extracted feature matrix of shape (%d, %d)", n, d)
    return feature_matrix, labels.astype(np.int64), list(FEATURE_NAMES)
