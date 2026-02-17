#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot power spectrum of occipital areas to check quality

Federico Ramírez-Toraño
17/02/2026

"""

# Imports
import mne
import numpy as np
from mpmath.functions.functions import powm1
from scipy.signal import welch
import matplotlib.pyplot as plt
from sympy.core.power import power

from test.init.init import init

from sEEGnal.tools.mne_tools import prepare_eeg
from sEEGnal.tools.bids_tools import build_BIDS_object, read_inverse_solution, read_forward_model
from sEEGnal.sources_reconstruction.atlas import label_aal


# Init the database
config, files, sub, ses, task = init()

# Go through each subject
index = range(len(files))
for current_index in index:

    # current info
    current_file = files[current_index]
    current_sub = sub[current_index]
    current_ses = ses[current_index]
    current_task = task[current_index]

    print(current_file)

    # Create the subjects following AI-Mind protocol
    BIDS = build_BIDS_object(config, current_sub, current_ses, current_task)

    # Add the subsystem info
    config['subsystem'] = 'source_reconstruction'

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

    raw = prepare_eeg(
        config,
        BIDS,
        raw=raw,
        apply_sobi=sobi,
        freq_limits=[2, 45],
        metadata_badchannels=True,
        epoch_definition=epoch_definition
    )

    # Get the atlas-sources information
    forward_model = read_forward_model(config,BIDS)
    atlas = label_aal(forward_model['src'])

    # Read LCMV beamformer
    lcmv_filters = read_inverse_solution(config,BIDS)

    # Apply filters
    stc = mne.beamformer.apply_lcmv_epochs(raw, lcmv_filters)

    # Get the sources of interest (occipital)
    occipital_source_idx = range(49,55)
    occipital_ts = [
        stc_epoch.data[occipital_source_idx].mean(axis=0)
        for stc_epoch in stc
    ]

    # Compute PSD
    sfreq = raw.info['sfreq']
    all_pow = []
    for ts in occipital_ts:
        f, pow = welch(ts,
                       fs=sfreq,
                       nperseg=sfreq * 2,
                       scaling='spectrum'
                       )
        all_pow.append(pow)

    # Get the mean and std of pow spectrum
    all_pow = np.array(all_pow)
    pow_mean = all_pow.mean(axis=0)
    pow_std = all_pow.std(axis=0)

    # Plot between 2 - 45 Hz
    freq_mask = (f >= 2) & (f <= 45)
    plt.figure(figsize=(8, 4))
    plt.plot(f[freq_mask], pow_mean[freq_mask], color='blue', label='Mean PSD')
    plt.fill_between(f[freq_mask],
                     pow_mean[freq_mask] - pow_std[freq_mask],
                     pow_mean[freq_mask] + pow_std[freq_mask],
                     color='skyblue', alpha=0.3)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title("Occipital Lobe Power Spectrum (2–45 Hz, Averaged Across Epochs)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)



