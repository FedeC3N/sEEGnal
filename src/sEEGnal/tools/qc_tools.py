"""
Quality check utilities for sEEGnal

Federico Ramírez-Toraño
06/04/2026
"""

# Imports
import os
import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from sEEGnal.tools.mne_tools import prepare_eeg
from sEEGnal.tools.bids_tools import build_derivatives_path


### ARTIFACTS QC
def artifact_qc(config, BIDS):

    # Check whether the occipital power spectrum looks correct
    plot_occipital_power_spectrum(config, BIDS)


def plot_occipital_power_spectrum(config, BIDS):

    # Load the clean EEG
    sobi = {
        'desc': 'sobi',
        'components_to_include': ['brain', 'other'],
        'components_to_exclude': []
    }

    freq_limits = [
        config['component_estimation']['low_freq'],
        config['component_estimation']['high_freq']
    ]
    crop_seconds = config['component_estimation']['crop_seconds']
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]
    epoch_definition = config['source_reconstruction']['epoch_definition']

    # Load the clean data
    raw = prepare_eeg(
        config,
        BIDS,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        freq_limits=freq_limits,
        notch_filter=True,
        resample_frequency=resample_frequency,
        set_annotations=True,
        crop_seconds=crop_seconds,
        rereference='average'
    )

    epochs = prepare_eeg(
        config,
        BIDS,
        raw=raw,
        apply_sobi=sobi,
        freq_limits=[2, 45],
        metadata_badchannels=True,
        exclude_badchannels=True,
        epoch_definition=epoch_definition
    )

    epochs.load_data()

    # Get occipital channels based on their position
    montage = epochs.get_montage()
    if montage is None:
        raise RuntimeError('No montage available to select occipital channels.')

    ch_pos = montage.get_positions()['ch_pos']
    available_channels = [ch for ch in epochs.ch_names if ch in ch_pos]

    if len(available_channels) == 0:
        raise RuntimeError('No channel positions available for the selected channels.')

    pos = np.array([ch_pos[ch] for ch in available_channels])

    # Posterior sensors in head coordinates (meters)
    occipital_idx = np.where(pos[:, 1] < -0.06)[0]
    picks = [available_channels[i] for i in occipital_idx]

    # Fallback: take the 8 most posterior channels
    if len(picks) == 0:
        n_keep = min(8, len(available_channels))
        occipital_idx = np.argsort(pos[:, 1])[:n_keep]
        picks = [available_channels[i] for i in occipital_idx]

    epochs_occ = epochs.copy().pick(picks)

    # Estimate spectrum
    spectrum = epochs_occ.compute_psd(
        method='welch',
        fmin=2,
        fmax=45,
    )

    freqs = spectrum.freqs.copy()
    psd = spectrum.get_data().copy()   # shape: (n_epochs, n_channels, n_freqs)

    # Convert PSD to relative power spectrum
    delta_f = freqs[1] - freqs[0]
    power_spectrum = psd * delta_f
    relative_power_spectrum = power_spectrum / power_spectrum.sum(axis=-1, keepdims=True)

    # Average across epochs -> shape: (n_channels, n_freqs)
    relative_power_spectrum_epochs_mean = np.mean(relative_power_spectrum, axis=0)

    # Mean and std across channels -> shape: (n_freqs,)
    relative_power_spectrum_mean = np.mean(relative_power_spectrum_epochs_mean, axis=0)
    power_spectrum_std = np.std(relative_power_spectrum_epochs_mean, axis=0)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(freqs, relative_power_spectrum_mean, label='Mean power spectrum')
    plt.fill_between(
        freqs,
        relative_power_spectrum_mean - power_spectrum_std,
        relative_power_spectrum_mean + power_spectrum_std,
        alpha=0.3
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title(f"Occipital Power Spectrum (averaged across {len(picks)} channels)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save the figure
    process = 'check'
    tail = 'occipital_power_spectrum'
    figure_path = build_derivatives_path(BIDS, process, tail)

    if not os.path.exists(figure_path.parent):
        os.makedirs(figure_path.parent)

    plt.savefig(figure_path)
    plt.close()