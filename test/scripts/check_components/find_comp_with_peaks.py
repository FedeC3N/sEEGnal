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
        freq_limits=freq_limits,
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
    ICs_time_series.pick(sobi.labels_['other'])

    # Estimate the power
    spectrum = ICs_time_series.compute_psd(
        method='welch',
        fmin=2,
        fmax=45,
        picks='all'
    )
    freqs = spectrum.freqs
    spectrum = spectrum.average().get_data(picks='all')

    # Create the figures to plot
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1,
                             figsize=(16, 8),
                             constrained_layout=True)


    peaks_diff = []
    for ispectrum in range(spectrum.shape[0]):

        current_spectrum = spectrum[ispectrum,:]
        current_peaks, _ = find_peaks(current_spectrum,prominence=0.0001)
        current_ax = axes[0]
        current_ax.plot(freqs, current_spectrum)
        current_ax.plot(freqs[current_peaks], current_spectrum[current_peaks],
                      '*r')

        if len(current_peaks) > 2:
            dummy = np.std(np.diff(current_peaks))
            peaks_diff.append(dummy)
        else:
            peaks_diff.append(np.nan)

    current_ax = axes[1]
    current_ax.plot(peaks_diff, '*')


    bads = [i for i in range(len(peaks_diff)) if peaks_diff[i] < 0.05]
    if len(bads) > 0:
        for ibad in bads:

            current_spectrum = spectrum[ibad, :]

            current_peaks, _ = find_peaks(current_spectrum,prominence=0.0001)

            current_ax = axes[2]
            current_ax.plot(freqs, current_spectrum)
            current_ax.plot(freqs[current_peaks], current_spectrum[current_peaks],'*r')

        peaks_diff = np.asarray(peaks_diff)
        bads = np.array(bads)
        current_ax = axes[1]
        current_ax.plot(bads,peaks_diff[bads],'r*')

    plt.show(block=True)