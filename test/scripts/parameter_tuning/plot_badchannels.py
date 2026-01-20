#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots for each criterion

Federico Ramírez-Toraño
20/01/2026

"""

# Imports
from sEEGnal.tools.mne_tools import prepare_eeg


def impossible_amplitude_detection(config,bids_path,badchannels):

    # Load the data
    raw = prepare_eeg(
            config,
            bids_path,
            preload=True,
            channels_to_include=config['global']["channels_to_include"],
            channels_to_exclude=config['global']["channels_to_exclude"],
            notch_filter=True,
            freq_limits=[1,100],
            resample_frequency=500,
            crop_seconds=10
        )

    # Add badchannels
    raw.info['bads'] = badchannels

    # Plot
    raw.plot(
        block=True,
        scalings=dict(eeg=50e-6))



def component_detection(config,bids_path,badchannels):

    # Load the data
    raw = prepare_eeg(
            config,
            bids_path,
            preload=True,
            channels_to_include=config['global']["channels_to_include"],
            channels_to_exclude=config['global']["channels_to_exclude"],
            notch_filter=True,
            freq_limits=[2,45],
            resample_frequency=500,
            crop_seconds=10
        )

    # Plot
    raw.plot(
        block=False,
        scalings=dict(eeg=50e-6))

    # Parameters for loading EEG  recordings
    sobi = {
        'desc': 'sobi-badchannels',
        'components_to_include': ['other','ch_noise','line_noise'],
        'components_to_exclude': []
    }
    freq_limits = [
        config['badchannel_detection']['component_detection']['low_freq'],
        config['badchannel_detection']['component_detection']['high_freq']
    ]
    channels_to_include = config['global']['channels_to_include']
    channels_to_exclude = config['global']['channels_to_exclude']
    resample_frequency = config['component_estimation']['resample_frequency']
    crop_seconds = config['badchannel_detection']['crop_seconds']

    # Load the raw EEG
    raw = prepare_eeg(
        config,
        bids_path,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        resample_frequency=resample_frequency,
        notch_filter=True,
        crop_seconds=crop_seconds
    )

    # Apply SOBI
    raw = prepare_eeg(
        config,
        bids_path,
        raw=raw,
        apply_sobi=sobi,
        freq_limits=freq_limits
    )

    # Add badchannels
    raw.info['bads'] = badchannels

    # Plot
    raw.plot(
        block=True,
        scalings=dict(eeg=10e-6))



def gel_bridge_detection(config, bids_path, badchannels):

    # Load the data
    raw = prepare_eeg(
            config,
            bids_path,
            preload=True,
            channels_to_include=config['global']["channels_to_include"],
            channels_to_exclude=config['global']["channels_to_exclude"],
            notch_filter=True,
            freq_limits=[2,45],
            resample_frequency=500,
            crop_seconds=10
        )

    # Plot only the gel bridged ones
    if len(badchannels) > 0:
        raw.pick(badchannels)

    # Plot
    raw.plot(
        block=True,
        butterfly=True,
        scalings=dict(eeg=50e-6))



def high_deviation_detection(config, bids_path, badchannels):
    # Parameters for loading EEG  recordings
    sobi = {
        'desc': 'sobi-badchannels',
        'components_to_include': ['brain', 'other'],
        'components_to_exclude': []
    }
    freq_limits = [
        config['badchannel_detection']['high_deviation']['low_freq'],
        config['badchannel_detection']['high_deviation']['high_freq']
    ]
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']['channels_to_include']
    channels_to_exclude = config['global']['channels_to_exclude']
    crop_seconds = config['badchannel_detection']['crop_seconds']

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
    )

    # Add badchannels
    raw.info['bads'] = badchannels

    # Plot
    raw.plot(
        block=True,
        scalings=dict(eeg=50e-6))



def all(config, bids_path, badchannels):

    # Parameters for loading EEG  recordings
    sobi = {
        'desc': 'sobi-badchannels',
        'components_to_include': ['brain', 'other'],
        'components_to_exclude': []
    }
    freq_limits = [
        config['badchannel_detection']['high_deviation']['low_freq'],
        config['badchannel_detection']['high_deviation']['high_freq']
    ]
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']['channels_to_include']
    channels_to_exclude = config['global']['channels_to_exclude']
    crop_seconds = config['badchannel_detection']['crop_seconds']

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
    )

    # Add badchannels
    raw.info['bads'] = badchannels

    # Plot
    raw.plot(
        block=True,
        scalings=dict(eeg=50e-6))