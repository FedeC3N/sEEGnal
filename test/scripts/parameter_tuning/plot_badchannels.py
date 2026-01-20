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

    # Plot
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



def power_spectrum_detection(config,bids_path,badchannels):

    # Plot
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
        block=False,
        scalings=dict(eeg=50e-6))

    # Plot
    # Load the data
    raw = prepare_eeg(
        config,
        bids_path,
        preload=True,
        channels_to_include=config['global']["channels_to_include"],
        channels_to_exclude=config['global']["channels_to_exclude"],
        notch_filter=True,
        freq_limits=[45, 55],
        resample_frequency=500,
        crop_seconds=10
    )

    # Add badchannels
    raw.info['bads'] = badchannels

    # Plot
    raw.plot(
        block=True,
        scalings=dict(eeg=10e-6))