#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimate different covariance matrix

Federico Ramírez-Toraño
11/02/2026

"""

# Imports
import sys

import mne

from sEEGnal.tools.mne_tools import prepare_eeg

def estimate_inverse_solution(config,bids_path,
                              forward_model=None,
                              data_cov=None,
                              noise_cov=None):
    """

    Helper to select the correct function

    """

    # Add the subsystem info
    config['subsystem'] = 'source_reconstruction'

    # Build the function name
    func = f"estimate_{config['source_reconstruction']['inverse']['method']}"
    to_estimate = getattr(sys.modules[__name__], func)

    # Estimate the inverse solution using the defined method
    filters = to_estimate(config,bids_path,
                          forward_model=forward_model,
                          data_cov=data_cov,
                          noise_cov=noise_cov)

    return filters


def estimate_LCMV(config,bids_path,
                  forward_model=None,
                  data_cov=None,
                  noise_cov=None):
    """

    Estimate LCMV beamformer filters

    """

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

    raw = prepare_eeg(
        config,
        bids_path,
        raw=raw,
        apply_sobi=sobi,
        freq_limits=[2, 45],
        metadata_badchannels=True,
        epoch_definition=epoch_definition
    )

    # Estimate the rank of the covariance matrix
    rank = mne.compute_rank(data_cov,info=raw.info)

    # Estimate the LCMV beamformers filters
    filters = mne.beamformer.make_lcmv(
        info=raw.info,
        forward=forward_model,
        data_cov=data_cov,
        reg=config['source_reconstruction']['inverse']['reg'],
        noise_cov=None,  # resting-state
        pick_ori=config['source_reconstruction']['inverse']['pick_ori'],
        weight_norm=config['source_reconstruction']['inverse']['weight_norm'],
        reduce_rank=True,
        rank=rank
    )

    return filters
