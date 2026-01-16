#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to load clean EEGs, estimate the power and plot the power spectrum

Federico Ramírez-Toraño
12/11/2025

"""

# Imports
import matplotlib.pyplot as plt

from init import init
from sEEGnal.tools.mne_tools import prepare_eeg
from sEEGnal.tools.bids_tools import create_bids_path, read_sobi

# Init the database
config, files, sub, ses, task = init()

# current info
subject_index = 10
current_file = files[subject_index]
current_sub = sub[subject_index]
current_ses = ses[subject_index]
current_task = task[subject_index]

# Create the subjects following AI-Mind protocol
bids_path = create_bids_path(config, current_sub, current_ses, current_task)

# Parameters to load the data
sobi                = {
        'desc': 'sobi',
        'components_to_include': ['brain','other'],
        'components_to_exclude': []
    }
freq_limits         = [
        config['component_estimation']['low_freq'],
        config['component_estimation']['high_freq']
]
crop_seconds = config['component_estimation']['crop_seconds']
resample_frequency = config['component_estimation']['resampled_frequency']
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
    metadata_badchannels=True
)

# Plot the clean data
fig = clean_data.plot(block=False,duration=20)
fig.canvas.manager.set_window_title(current_file)

# Remove the bad epochs before estimating power spectrum
clean_data = prepare_eeg(
    config,
    bids_path,
    raw=clean_data,
    preload=True,
    epoch=epoch_definition
)
clean_data.drop_bad()

# Estimate the power
spectrum = clean_data.compute_psd(
    method='welch',
    fmin=2,
    fmax=45,
)
spectrum = spectrum.average()
fig = spectrum.plot(dB=False, amplitude=True)
fig.canvas.manager.set_window_title(current_file)
plt.show(block=True)
