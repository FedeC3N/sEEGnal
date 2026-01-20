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

from sEEGnal.tools.mne_tools import prepare_eeg


def EOG_detection(config, bids_path, annotations):

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
        bids_path,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        notch_filter=True,
        resample_frequency=resample_frequency,
        crop_seconds=crop_seconds
    )

    # Apply SOBI
    raw = prepare_eeg(
        config,
        bids_path,
        raw=raw,
        apply_sobi=sobi
    )

    # Filter
    raw = prepare_eeg(
        config,
        bids_path,
        raw=raw,
        preload=True,
        freq_limits=freq_limits,
        rereference='average'
    )

    # Add annotations
    raw.set_annotations(annotations)

    # Plot
    raw.plot(
        block=False,
        scalings=dict(eeg=50e-6))

    # Filter
    raw.filter(0.5,2)

    # Get the frontal and background channels
    frontal_channels = config['artifact_detection']['frontal_channels']
    background_channels = [
        current_channel for current_channel in raw.ch_names if
        current_channel not in frontal_channels
    ]

    # Create the average
    frontal_raw = raw.copy().pick_channels(frontal_channels).get_data()
    frontal_avg = np.mean(frontal_raw,axis=0,keepdims=True)
    background_raw = raw.copy().pick_channels(background_channels).get_data()
    background_avg = np.mean(background_raw, axis=0, keepdims=True)
    avg_data = np.concatenate((frontal_avg, background_avg), axis=0)

    # Create Raw object
    info = mne.create_info(
        ch_names=['frontal_avg','background_avg'],
        sfreq=raw.info['sfreq'],
        ch_types=['eeg','eeg']
    )

    # Create new Raw object
    raw_avg = mne.io.RawArray(avg_data, info)

    # Plot
    raw_avg.plot(
        block=True,
        scalings=dict(eeg=50e-6))