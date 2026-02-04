#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot EEG power spectrum

Federico Ramírez-Toraño
04/02/2026

"""

# Import
import mne
import numpy as np
import matplotlib.pyplot as plt

from sEEGnal.tools.mne_tools import prepare_eeg
from sEEGnal.tools.bids_tools import read_sobi


def plot_component_power(config, bids_path):

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

    # Get the common figure
    fig, axes = plt.subplots(2, 4,
                             figsize=(16,8),
                             constrained_layout=True)
    fig.canvas.manager.set_window_title(bids_path.basename)
    axes = axes.flatten()

    # Plot the EEG spectrum
    plot_eeg_power(config,bids_path,axes[0])

    # Plot IC classification
    IClabel_componets = ['brain', 'muscle', 'eog', 'ecg', 'line_noise',
                         'ch_noise', 'other']

    # Plots
    for current_category, current_ax in zip(IClabel_componets,axes[1:]):

        # Print the values
        component_index = sobi.labels_[current_category]
        dummy = [x + 1 for x in component_index]
        dummy = np.array(dummy).tolist()
        print(f"    {current_category}: {dummy}")

        if len(component_index) > 0 :

            current_IC_time_series = ICs_time_series.copy().pick(component_index)

            # Estimate the power
            spectrum = current_IC_time_series.compute_psd(
                method='welch',
                fmin=2,
                fmax=45,
                picks='all'
            )
            spectrum = spectrum.average()

            # Plot
            spectrum.plot(
                dB=False,
                amplitude=True,
                picks='all',
                axes=current_ax)

        # Add info
        current_ax.set_title(current_category)

    # Add extra space to the screen
    print()
    print()

    # Show
    plt.show(block=True)




def plot_eeg_power(config, bids_path,ax):

# Parameters to load the data
    sobi                = {
            'desc': 'sobi',
            'components_to_include': ['brain'],
            'components_to_exclude': []
        }
    freq_limits         = [
            config['component_estimation']['low_freq'],
            config['component_estimation']['high_freq']
    ]
    crop_seconds = config['component_estimation']['crop_seconds']
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]
    epoch_definition = {"length": 4, "overlap": 0, "padding": 0}


    # Load the clean data
    clean_data = prepare_eeg(
        config,
        bids_path,
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

    # Apply SOBI
    clean_data = prepare_eeg(
        config,
        bids_path,
        raw=clean_data,
        apply_sobi=sobi,
        freq_limits=[2, 45],
        metadata_badchannels=True,
        exclude_badchannels=True,
        epoch_definition=epoch_definition
    )

    # Estimate the power
    spectrum = clean_data.compute_psd(
        method='welch',
        fmin=2,
        fmax=45,
    )
    spectrum = spectrum.average()
    spectrum.plot(dB=False, amplitude=True,show=False,axes=ax)