#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The noisy component has several peaks

Federico Ramírez-Toraño
04/02/2026

"""

# Imports
import numpy as np
from scipy.signal import find_peaks

from test.init.init import init
from sEEGnal.tools.mne_tools import prepare_eeg
from sEEGnal.tools.bids_tools import create_bids_path, read_sobi


# Init the database
config, files, sub, ses, task = init()

# Random order to avoid seen the same subjects over and over again
index = range(len(files))
index = np.random.permutation(len(files))
for subject_index in index:

    current_file = files[subject_index]
    current_sub = sub[subject_index]
    current_ses = ses[subject_index]
    current_task = task[subject_index]

    # Create the subjects following AI-Mind protocol
    bids_path = create_bids_path(config, current_sub, current_ses, current_task)

    print(bids_path.basename)

    # Parameters for loading EEG recordings
    freq_limits = [
        config['component_estimation']['low_freq'],
        config['component_estimation']['high_freq']
    ]
    crop_seconds = config['component_estimation']['crop_seconds']
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]
    epoch_definition = config['component_estimation']['epoch_definition']
    set_annotations = True

    # Load raw EEG
    raw = prepare_eeg(
        config,
        bids_path,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        notch_filter=True,
        freq_limits=[6,45],
        resample_frequency=resample_frequency,
        metadata_badchannels=True,
        interpolate_badchannels=True,
        set_annotations=set_annotations,
        crop_seconds=crop_seconds,
        rereference='average',
        epoch_definition=epoch_definition
    )

    # Read the ICA information
    sobi = read_sobi(bids_path, 'sobi')
    ICs_time_series = sobi.get_sources(raw)
    ICs_time_series = ICs_time_series.pick(sobi.labels_['other'])

    # Estimate the power
    spectrum = ICs_time_series.compute_psd(
        method='welch',
        fmin=6,
        fmax=45,
        picks='all'
    )
    freqs = spectrum.freqs
    spectrum = spectrum.average().get_data(picks='all')

    import matplotlib.pyplot as plt
    for ispectrum in range(spectrum.shape[0]):

        current_spectrum = spectrum[ispectrum, :]
        current_peaks, _ = find_peaks(
            current_spectrum,
            prominence=0.00001)

        if len(current_peaks) > 3:


            # Check
            current_peaks = np.array(current_peaks)
            k = np.arange(len(current_peaks))
            d_hat, p0_hat = np.polyfit(k, current_peaks, 1)
            residuals = current_peaks - (p0_hat + k * d_hat)
            spacing_score = np.std(residuals)
            tol = 1
            is_periodic = np.all(np.abs(residuals) < tol)

            if is_periodic:
                fig, axes = plt.subplots(2, 1,
                                         figsize=(16, 8),
                                         constrained_layout=True)
                axes[0].plot(freqs, current_spectrum,'-*')
                axes[0].plot(freqs[current_peaks], current_spectrum[current_peaks],
                         '*r')

                fitted_line = p0_hat + d_hat * k
                axes[1].plot(k, current_peaks, 'o', label='Picos detectados',
                         markersize=8)
                axes[1].plot(k, fitted_line, '-',
                         label=f'Ajuste lineal: d_hat={d_hat:.2f}')
                plt.show(block=True)


