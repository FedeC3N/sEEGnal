# -*- coding: utf-8 -*-
"""
Spectral estimation utilities for sEEGnal

Federico Ramírez-Toraño (and ChatGPT 5-mini)
26/02/2026
"""

import numpy as np
from scipy.signal.windows import dpss
from scipy.fft import rfft, rfftfreq


# ==========================================================
# DPSS
# ==========================================================

def compute_dpss(n_times, sfreq, bandwidth, low_bias=True):
    """
    Compute DPSS tapers.

    Parameters
    ----------
    n_times : int
    sfreq : float
    bandwidth : float
        Frequency smoothing (Hz)
    low_bias : bool
        Keep only tapers with eigenvalue > 0.9

    Returns
    -------
    tapers : ndarray (n_tapers, n_times)
    eigvals : ndarray (n_tapers,)
    """
    NW = bandwidth * n_times / (2 * sfreq)
    Kmax = int(2 * NW)
    tapers, eigvals = dpss(n_times, NW, Kmax, return_ratios=True)
    if low_bias:
        mask = eigvals > 0.9
        tapers = tapers[mask]
        eigvals = eigvals[mask]
    return tapers, eigvals


# ==========================================================
# Multitaper PSD (epoch-by-epoch, taper-by-taper)
# ==========================================================

def multitaper_psd(
    data,
    sfreq,
    fmin=0.0,
    fmax=np.inf,
    bandwidth=4.0,
    adaptive=False,
    low_bias=True,
    scaling="density",
    dtype=np.float32
):
    """
    Compute multitaper PSD per epoch (memory-efficient).

    Parameters
    ----------
    data : ndarray
        Shape:
            (n_signals, n_times)
            OR
            (n_epochs, n_signals, n_times)
    sfreq : float
    fmin, fmax : float
    bandwidth : float
    adaptive : bool
        Use Thomson adaptive weighting
    low_bias : bool
    scaling : str
        "density" or "spectrum"
    dtype : numpy dtype
        Data type to reduce memory (float32 recommended)

    Returns
    -------
    psd : ndarray (n_epochs, n_signals, n_freqs)
    freqs : ndarray (n_freqs,)
    """
    data = np.asarray(data, dtype=dtype)
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    n_epochs, n_signals, n_times = data.shape

    # DPSS tapers
    tapers, eigvals = compute_dpss(n_times, sfreq, bandwidth, low_bias)
    n_tapers = len(tapers)

    # Frequency vector
    freqs = rfftfreq(n_times, 1 / sfreq)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[freq_mask]
    n_freqs = len(freqs)

    # Output PSD
    psd = np.zeros((n_epochs, n_signals, n_freqs), dtype=dtype)

    # Loop over epochs to reduce memory
    for e in range(n_epochs):
        # PSD per taper
        taper_psds = np.zeros((n_signals, n_tapers, n_freqs), dtype=dtype)

        for k in range(n_tapers):
            tapered = data[e] * tapers[k]  # n_signals x n_times
            fft_data = rfft(tapered, axis=-1)[:, freq_mask]

            # One-sided correction
            fft_data[:, 1:-1] *= np.sqrt(2.0)

            # Power scaling
            if scaling == "density":
                power = np.abs(fft_data) ** 2 / (sfreq * n_times)
            elif scaling == "spectrum":
                power = np.abs(fft_data) ** 2 / n_times
            else:
                raise ValueError("scaling must be 'density' or 'spectrum'")

            taper_psds[:, k, :] = power

        # Adaptive weighting
        if adaptive:
            epoch_psd = _adaptive_weighting(taper_psds, eigvals)
        else:
            epoch_psd = taper_psds.mean(axis=1)

        psd[e] = epoch_psd

    return psd, freqs


# ==========================================================
# Thomson Adaptive Weighting
# ==========================================================

def _adaptive_weighting(psd, eigvals, max_iter=50, tol=1e-10):
    """
    Thomson adaptive weighting.

    Parameters
    ----------
    psd : ndarray (n_signals, n_tapers, n_freqs)
    eigvals : ndarray (n_tapers,)
    """
    n_signals, n_tapers, n_freqs = psd.shape
    S = psd.mean(axis=1)  # initial estimate n_signals x n_freqs

    eigvals = eigvals[np.newaxis, :, np.newaxis]  # 1 x n_tapers x 1

    for _ in range(max_iter):
        S_prev = S.copy()
        weights = (eigvals * S[:, np.newaxis, :]) / (
            eigvals * S[:, np.newaxis, :] + (1 - eigvals) * psd
        )
        weights /= np.sum(weights, axis=1, keepdims=True)
        S = np.sum(weights * psd, axis=1)
        if np.max(np.abs(S - S_prev)) < tol:
            break

    return S


# ==========================================================
# PSD Normalization
# ==========================================================

def normalize_psd(psd, mode="relative"):
    """
    Normalize PSD per epoch/signal.

    Parameters
    ----------
    psd : ndarray
        Shape (n_epochs, n_signals, n_freqs)
    mode : str
        "relative", "log", or None

    Returns
    -------
    psd_norm : ndarray
    """
    if mode is None:
        return psd
    if mode == "relative":
        total_power = np.sum(psd, axis=-1, keepdims=True)
        total_power[total_power == 0] = np.finfo(float).eps
        return psd / total_power
    if mode == "log":
        return np.log10(psd + np.finfo(float).eps)
    raise ValueError("mode must be None, 'relative', or 'log'")