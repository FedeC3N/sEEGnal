#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep changes in prepare_eeg

Federico Ramírez-Toraño
11/12/2025

"""

# Imports
import numpy as np

from init import init
from sEEGnal.tools.bids_tools import build_BIDS, read_badchannels, read_annotations
from sEEGnal.tools.mne_tools import prepare_eeg


# Init the database
config, files, sub, ses, task = init()

# current info
current_file = files[0]
current_sub = sub[0]
current_ses = ses[0]
current_task = task[0]

# Create the subjects following AI-Mind protocol
BIDS = build_BIDS(config, current_sub, current_ses, current_task)

##################################################################
# Check include / exclude channels
##################################################################
'''
print('Check include / exclude channels')

# Exclusion
raw = prepare_eeg(
    config,
    BIDS
)
channels_all = raw.ch_names

channels_to_exclude = ['Fp1']
raw = prepare_eeg(
    config,
    BIDS,
    channels_to_exclude=channels_to_exclude
)
channels_with_exclusion = raw.ch_names
dummy1 = set(channels_all) - set(channels_to_exclude)
dummy2 = set(channels_with_exclusion)
if len(dummy1-dummy2) == 0:
    print('  Channel exclusion OK')
else:
    print('  Channel exclusion FAILED')

# Inclusion
channels_to_include = ['Fp1']
raw = prepare_eeg(
    config,
    BIDS,
    channels_to_include=channels_to_include
)
channels_with_inclusion = raw.ch_names
dummy1 = set(channels_to_include)
dummy2 = set(channels_with_inclusion)
if len(dummy1-dummy2) == 0:
    print('  Channel exclusion OK')
else:
    print('  Channel exclusion FAILED')
    
print()
'''

##################################################################
# Check include / exclude components
##################################################################
'''
print('Check include / exclude components')

# Create a SOBI dictionary
sobi = {
    'desc': 'sobi-badchannels',
    'components_to_include': [],
    'components_to_exclude': ['other'],
}
# Channels
channels_to_include = config['global']["channels_to_include"]
channels_to_exclude = config['global']["channels_to_exclude"]

raw = prepare_eeg(
    config,
    BIDS,
    preload=True,
    channels_to_include=channels_to_include,
    channels_to_exclude=channels_to_exclude,
    apply_sobi=sobi
)
raw.plot(block=True, duration=20)
print()
'''

##################################################################
# Check resample frequency
##################################################################
'''
print('Check resample frequency')

# Get the original frequency
raw = prepare_eeg(
    config,
    BIDS,
    preload=True
)
original_fs = raw.info['sfreq']

# Downsample
raw = prepare_eeg(
    config,
    BIDS,
    preload=True,
    resample_frequency=250
)
resampled_fs = raw.info['sfreq']

# Check
if original_fs != resampled_fs:
    print('  Resample OK')
else:
    print('  Resample FAILED')

print()
'''

##################################################################
# Check filters
##################################################################
'''
print('Check filters')

freq_limits = [2,45]
raw = prepare_eeg(
    config,
    BIDS,
    preload=True,
    freq_limits=freq_limits,
    notch_filter=True
)

# Check the filters
if (freq_limits[0] == raw.info['highpass']) and (freq_limits[1] == raw.info['lowpass']):
    print('  Filter OK')
else:
    print('  Filter FAILED')

print()
'''

##################################################################
# Check exclude bad channels
##################################################################
'''
print('Check exclude bad channels')

# Get the original number of channels and bad channels
channels_data = read_chan(BIDS)
all_channels = len(list(channels_data['name']))
badchannels = len((list(channels_data.loc[channels_data['status'] == 'bad']['name'])))

# Exclude bad channels
raw = prepare_eeg(
    config,
    BIDS,
    preload=True,
    exclude_badchannels=True,
    interpolate_badchannels=False
)
exclude_channels = len(raw.info['ch_names'])

# Interpolate bad channels
raw = prepare_eeg(
    config,
    BIDS,
    preload=True,
    exclude_badchannels=True,
    interpolate_badchannels=True
)
interpolate_channels = len(raw.info['ch_names'])

if (all_channels == exclude_channels + badchannels) and (all_channels == interpolate_channels):
    print('  Exclude badchannels OK')
else:
    print('  Exclude badchannels FAILED')
    '''


##################################################################
# Check artefacts
##################################################################
'''
print('Check artefacts loading')

# Get the original number of artefacts, types, duration, and onset
annotations_read = read_annot(BIDS)
num_of_artefacts = len(annotations_read)

# Load the EEG recording with the annotations
sobi = {
    'desc': 'sobi',
    'components_to_include': ['brain'],
    'components_to_exclude': []
}
raw = prepare_eeg(
    config,
    BIDS,
    apply_sobi=sobi,
    notch_filter=True,
    freq_limits=[2,45],
    set_annotations=True,
)
annotations_load = raw.annotations

# Check if correctly read
test_ok = True
if len(annotations_read) != len(annotations_load):
    test_ok = False
for i in range(max(len(annotations_read) ,len(annotations_load))):
    if annotations_read.description[i] != annotations_load.description[i]:
        test_ok = False
    if annotations_read.duration[i] != annotations_load.duration[i]:
        test_ok = False
    if annotations_read.onset[i] != annotations_load.onset[i]:
        test_ok = False

# Check the same when cropped
# Load the EEG recording with the annotations
sobi = {
    'desc': 'sobi',
    'components_to_include': ['brain'],
    'components_to_exclude': []
}
raw_crop = prepare_eeg(
    config,
    BIDS,
    apply_sobi=sobi,
    notch_filter=True,
    freq_limits=[2,45],
    set_annotations=True,
    crop_seconds=50
)

# Plot to see if they are in the same place
raw.plot(block=False, duration=10)
raw_crop.plot(block=True, duration=10)

# Print results
if test_ok:
    print('  Load artefacts OK')
else:
    print('  Load artefacts FAILED')
'''



