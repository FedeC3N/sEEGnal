#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimate different covariance matrix

Federico Ramírez-Toraño
11/02/2026

"""

# Imports
import sys
import traceback
from datetime import datetime as dt, timezone

import mne

from sEEGnal.tools.mne_tools import prepare_eeg
from sEEGnal.tools.bids_tools import read_forward_model, write_inverse_solution



def estimate_inverse_solution(config, BIDS):
    """

    Helper to select the correct function

    """

    # Add the subsystem info
    config['subsystem'] = 'source_reconstruction'

    try:

        # Build the function name
        func = f"estimate_{config['source_reconstruction']['inverse']['method']}"
        to_estimate = getattr(sys.modules[__name__], func)

        # Estimate the inverse solution using the defined method
        inverse_solution = to_estimate(config, BIDS)

        # Save the metadata
        now = dt.now(timezone.utc)
        formatted_now = now.strftime("%d-%m-%Y %H:%M:%S")
        results = {
            'result': 'ok',
            'bids_basename': BIDS.basename,
            "date": formatted_now,
            'inverse_solution': inverse_solution
        }

    except Exception as e:

        # Save the error
        now = dt.now(timezone.utc)
        formatted_now = now.strftime("%d-%m-%Y %H:%M:%S")
        results = {
            'result': 'error',
            'bids_basename': BIDS.basename,
            "date": formatted_now,
            "details": f"Exception: {str(e)}, {traceback.format_exc()}"
        }

    return results


def estimate_lcmv(config, BIDS):
    """

    Estimate LCMV beamformer filters

    """

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

    # Estimate the data covariance
    data_cov = mne.compute_covariance(
        raw,
        method=config['source_reconstruction']['covariance']['method'],
        rank=config['source_reconstruction']['covariance']['rank']
    )

    # Estimate the rank of the covariance matrix
    rank = mne.compute_rank(data_cov,info=raw.info)

    # Load the forward model
    forward_model = read_forward_model(config,BIDS)

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

    # Save the inverse solution
    write_inverse_solution(config, BIDS, filters)

    return filters
