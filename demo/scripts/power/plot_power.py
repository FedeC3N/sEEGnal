#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to load clean EEGs, estimate the power and plot the power spectrum

Federico Ramírez-Toraño
12/11/2025

"""

# Imports
import os
import matplotlib.pyplot as plt

from sEEGnal.tools.mne_tools import prepare_eeg
from sEEGnal.tools.bids_tools import build_BIDS_object, read_sobi

# Select a subjet
config = {'path':{}}
config['path']['data_root']     = os.path.join('data')
current_sub                     = '003'
current_ses                     = '0'
current_task                    = '4EC'

# Get the BIDS path
BIDS = build_BIDS_object(config, current_sub, current_ses, current_task)

# Parameters to load the data
epoch_definition ={ "length": 4 , "overlap": 0 , "padding": 2,
                    'reject_by_annotation': 1 }
config = {'component_estimation':{}}
config['component_estimation']['notch_frequencies'] = [50, 100, 150, 200, 250]

# Load the clean data
clean_data = prepare_eeg(
        config,
        BIDS,
        preload=True,
        freq_limits=[2,45],
        crop_seconds=[10],
        resample_frequency=500,
        exclude_badchannels=True,
        set_annotations=True,
        rereference=True,
        interpolate_badchannels=True,
        epoch_definition=epoch_definition
        )

# Remove artefactual components
# Load the component label
sobi = read_sobi(config,BIDS,'sobi')

# Select the components of interest
components_to_include = []
if 'brain' in sobi.labels_.keys():
    components_to_include.append(sobi.labels_['brain'])
if 'other' in sobi.labels_.keys():
    components_to_include.append(sobi.labels_['other'])
components_to_include = sum(components_to_include, [])

# If desired components, apply them.
if len(components_to_include) > 0:
    # Remove the eog components
    sobi.apply(clean_data, include=components_to_include)

# Estimate the power
spectrum = clean_data.compute_psd(
    method  = 'welch',
    fmin    = 2,
    fmax    = 45,
)

spectrum = spectrum.average()
spectrum.plot(dB=False,amplitude=True)
plt.show(block=True)