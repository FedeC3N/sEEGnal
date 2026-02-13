#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the forward model

Federico Ramírez-Toraño
11/02/2026

"""

# Import
import mne

from sEEGnal.tools.mne_tools import prepare_eeg
from sEEGnal.tools.bids_tools import write_forward_model


def make_forward_model(config, BIDS):

    # Add the subsystem info
    config['subsystem'] = 'source_reconstruction'

    # Estimate the parameters of the forward model
    if config['source_reconstruction']['forward']['use_template']:
        forward_model = template_forward_model(config, BIDS)
    else:
        forward_model = []

    # Save the forward solution
    write_forward_model(config, BIDS, forward_model)

    return forward_model


def template_forward_model(config, bids_path):
    """

    Use the Freesurfer template to create a forward model

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
    )

    # Get the FreeSurfer fsaverage information
    fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    subject = config['source_reconstruction']['forward']['template']['subject']
    trans = config['source_reconstruction']['forward']['template']['trans']
    bem = fs_dir / "bem" / config['source_reconstruction']['forward']['template']['bem']

    # Define our sources
    src = mne.setup_volume_source_space(
        subject=subject,
        pos=config['source_reconstruction']['forward']['template']['pos'],
        mri=config['source_reconstruction']['forward']['template']['mri'],
        bem=None,
        add_interpolator=True
    )

    # Create the output forward model
    forward_model = mne.make_forward_solution(
        raw.info,
        trans,
        src,
        bem,
        meg=False,
        eeg=True,
        mindist=5.0
    )

    # Save the forward model
    return forward_model