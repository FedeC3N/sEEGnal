#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to test one by one the bad channel detection and artifact detection
criteria

Federico Ramírez-Toraño
19/01/2026

"""

# Imports
from test.init.init import init
import sEEGnal.preprocess.find_badchannels as fbc
from sEEGnal.tools.mne_tools import prepare_eeg
from sEEGnal.tools.bids_tools import create_bids_path

# Parameters
subject_index = 2
function = 'power_spectrum_detection'
sobi  = {
    'desc': 'sobi',
    'components_to_include': ['brain', 'other'],
    'components_to_exclude': []
}

# Init the database
config, files, sub, ses, task = init()
current_file = files[subject_index]
current_sub = sub[subject_index]
current_ses = ses[subject_index]
current_task = task[subject_index]

# Create the subjects following AI-Mind protocol
bids_path = create_bids_path(config, current_sub, current_ses, current_task)

# Call the specific function
func = getattr(fbc, function)
badchannels = func(config,bids_path)

# Plot
# Load the clean data
raw = prepare_eeg(
        config,
        bids_path,
        preload=True,
        channels_to_include=config['global']["channels_to_include"],
        channels_to_exclude=config['global']["channels_to_exclude"],
        freq_limits=[1,100],
        resample_frequency=500,
        crop_seconds=10,
        rereference='average'
    )

# Apply SOBI
raw = prepare_eeg(
    config,
    bids_path,
    raw=raw,
    apply_sobi=sobi,
    freq_limits=[2, 45],
    metadata_badchannels=True
)

# Plot
raw.plot(
    block=True,
    scalings=dict(eeg=50e-6))
