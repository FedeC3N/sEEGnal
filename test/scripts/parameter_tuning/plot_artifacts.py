#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots for each criterion

Federico Ramírez-Toraño
20/01/2026

"""

# Imports
import mne
import numpy as np

import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation

from sEEGnal.tools.mne_tools import prepare_eeg


def EOG_detection(config, BIDS):

    # Parameters for loading EEG  recordings
    sobi = {
        'desc': 'sobi',
        'components_to_include': ['brain', 'other'],
        'components_to_exclude': []
    }
    freq_limits = [
        config['visualization']['low_freq'],
        config['visualization']['high_freq']
    ]
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']['channels_to_include']
    channels_to_exclude = config['global']['channels_to_exclude']
    crop_seconds = config['component_estimation']['crop_seconds']

    # Load the raw EEG
    raw = prepare_eeg(
        config,
        BIDS,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        metadata_badchannels=True,
        interpolate_badchannels=True,
        notch_filter=True,
        resample_frequency=resample_frequency,
        set_annotations=True,
        crop_seconds=crop_seconds
    )

    # Apply SOBI
    raw = prepare_eeg(
        config,
        BIDS,
        raw=raw,
        apply_sobi=sobi
    )

    # Filter
    raw = prepare_eeg(
        config,
        BIDS,
        raw=raw,
        preload=True,
        freq_limits=freq_limits,
        rereference='average'
    )

    # Plot
    raw.plot(
        block=False,
        scalings=dict(eeg=50e-6))

    # Parameters for loading EEG recordings
    sobi = {
        'desc': 'sobi',
        'components_to_include': [],
        'components_to_exclude': ['eog', 'ecg']
    }
    freq_limits = [
        config['preprocess']['artifact_detection']['EOG']['low_freq'],
        config['preprocess']['artifact_detection']['EOG']['high_freq']
    ]
    crop_seconds = config['component_estimation']['crop_seconds']
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]
    epoch_definition = config['preprocess']['artifact_detection']['EOG']['epoch_definition']

    # Load the raw and apply SOBI
    raw = prepare_eeg(
        config,
        BIDS,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        resample_frequency=resample_frequency,
        metadata_badchannels=True,
        interpolate_badchannels=True,
        set_annotations=True,
        rereference='average'
    )

    # Apply SOBI and filters
    raw = prepare_eeg(
        config,
        BIDS,
        raw=raw,
        apply_sobi=sobi,
        freq_limits=freq_limits
    )

    # Select frontal channels
    frontal_channels = config['preprocess']['artifact_detection']['frontal_channels']
    frontal_channels = [
        current_channel for current_channel in frontal_channels if
        current_channel in channels_to_include
    ]
    if len(frontal_channels) > 0:
        frontal_raw = raw.copy()
        frontal_raw.pick(frontal_channels)
    else:
        return [], [], []  # If no frontal channels, no EOG artifacts

    raw_avg = mne.channels.combine_channels(
        raw,
        groups=dict(AVG=range(len(frontal_channels))),
        method='mean'
    )
    raw_avg.set_annotations(raw.annotations)
    raw_avg.crop(tmin=crop_seconds, tmax=raw_avg.times[-1] - crop_seconds)

    # Plot
    raw_avg.plot(
        block=True,
        scalings=dict(eeg=50e-6))


def muscle_detection(config,BIDS):

    # Plot the clean visualization
    # Parameters for loading EEG  recordings
    sobi = {
        'desc': 'sobi',
        'components_to_include': ['brain', 'other'],
        'components_to_exclude': []
    }
    freq_limits = [
        config['visualization']['low_freq'],
        config['visualization']['high_freq']
    ]
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']['channels_to_include']
    channels_to_exclude = config['global']['channels_to_exclude']
    crop_seconds = config['component_estimation']['crop_seconds']

    # Load the raw EEG
    raw = prepare_eeg(
        config,
        BIDS,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        metadata_badchannels=True,
        interpolate_badchannels=True,
        notch_filter=True,
        resample_frequency=resample_frequency,
        set_annotations=True,
        crop_seconds=crop_seconds
    )

    # Apply SOBI
    raw = prepare_eeg(
        config,
        BIDS,
        raw=raw,
        apply_sobi=sobi
    )

    # Filter
    raw = prepare_eeg(
        config,
        BIDS,
        raw=raw,
        preload=True,
        freq_limits=freq_limits,
        rereference='average'
    )

    # Plot
    raw.plot(
        block=False,
        scalings=dict(eeg=50e-6))


    # Plot the recording we have used to detect peaks
    # Parameters for loading EEG recordings
    freq_limits = [
        config['preprocess']['artifact_detection']['muscle']['low_freq'],
        config['preprocess']['artifact_detection']['muscle']['high_freq']
    ]
    crop_seconds = config['component_estimation']['crop_seconds']
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]

    # Load the raw and apply SOBI
    raw = prepare_eeg(
        config,
        BIDS,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        resample_frequency=resample_frequency,
        notch_filter=True,
        freq_limits=freq_limits,
        crop_seconds=crop_seconds,
        metadata_badchannels=True,
        exclude_badchannels=True,
        set_annotations=True
    )

    # Estimate the std across channels
    raw_std = np.std(raw.get_data(), axis=0)

    # Find peaks based on the total height (demeaning the signal first)
    raw_std = raw_std - np.mean(raw_std)
    height = (config['preprocess']['artifact_detection']['muscle']['threshold']
              * np.std(raw_std))
    muscle_index, _ = find_peaks(
        raw_std,
        height=height
    )

    plt.figure()
    plt.plot(
        raw.times, raw_std, 'b-',
        raw.times, height * np.ones(raw.times.shape), 'r--',
        raw.times[muscle_index], raw_std[muscle_index], 'r*'
    )
    raw.plot(
        block=True,
        scalings=dict(eeg=50e-6)
    )


def sensor_detection(config,BIDS):
    # Plot the clean visualization
    # Parameters for loading EEG  recordings
    sobi = {
        'desc': 'sobi',
        'components_to_include': ['brain', 'other'],
        'components_to_exclude': []
    }
    freq_limits = [
        config['visualization']['low_freq'],
        config['visualization']['high_freq']
    ]
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']['channels_to_include']
    channels_to_exclude = config['global']['channels_to_exclude']
    crop_seconds = config['component_estimation']['crop_seconds']

    # Load the raw EEG
    raw = prepare_eeg(
        config,
        BIDS,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        metadata_badchannels=True,
        interpolate_badchannels=True,
        notch_filter=True,
        resample_frequency=resample_frequency,
        set_annotations=True,
        crop_seconds=crop_seconds
    )

    # Apply SOBI
    raw = prepare_eeg(
        config,
        BIDS,
        raw=raw,
        apply_sobi=sobi
    )

    # Filter
    raw = prepare_eeg(
        config,
        BIDS,
        raw=raw,
        preload=True,
        freq_limits=freq_limits,
        rereference='average'
    )

    # Plot
    raw.plot(
        block=False,
        scalings=dict(eeg=50e-6))

    # Parameters for loading EEG recordings
    sobi = {
        'desc': 'sobi',
        'components_to_include': [],
        'components_to_exclude': ['eog', 'ecg']
    }
    freq_limits = [
        config['preprocess']['artifact_detection']['sensor']['low_freq'],
        config['preprocess']['artifact_detection']['sensor']['high_freq']
    ]
    crop_seconds = config['component_estimation']['crop_seconds']
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]


    # Load the raw data
    raw = prepare_eeg(
        config,
        BIDS,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        notch_filter=True,
        resample_frequency=resample_frequency,
        metadata_badchannels=True,
        interpolate_badchannels=True,
        set_annotations=True,
        crop_seconds=crop_seconds,
        rereference='average'
    )

    # Apply SOBI and filters
    raw = prepare_eeg(
        config,
        BIDS,
        raw=raw,
        preload=True,
        freq_limits=freq_limits,
        apply_sobi=sobi
    )

    raw.plot(
        block=True,
        scalings=dict(eeg=50e-6)
    )

    '''# Get the clean data
    raw_data = raw.get_data()

    # Get the maximum value for each channel and epoch
    max_values = np.max(np.abs(raw_data), axis=2, keepdims=False)

    # For each epoch, get the median and MAD across channels
    median = np.median(max_values, axis=1)
    MAD = median_abs_deviation(max_values, axis=1)
    threshold = (median + config['preprocess']['artifact_detection']['sensor']['threshold'] *
                 MAD)


    plt.plot(raw.events[:, 0] / 500 / 0.5, max_values)
    plt.scatter(raw.events[:, 0] / 500 / 0.5, threshold,
                c='r',
                marker='*')'''


