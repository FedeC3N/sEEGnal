#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script to obtain clean recordings from the metadata

Created on Thu 20/11/2024

@author: Fede
"""

# Imports
import os

import sEEGnal.tools.mnetools as aimind_mne
import sEEGnal.io.bids as bids
from eeglabio.utils import export_mne_epochs

def export_clean(config,bids_path):
    """

    For each recording, export it clean

    """

    if bids_path.datatype == 'eeg':

        # Create the derivatives bids_path
        out_filepath = bids.build_derivative(bids_path,'desc-sEEGnal_clean_eeg.set')

        # Check the folder
        if not os.path.exists(os.path.dirname(out_filepath)):
            os.makedirs(os.path.dirname(out_filepath))

        # Read the ICA information
        sobi = bids.read_sobi(bids_path, 'sobi')

        # Parameters
        channels_to_include = sobi.ch_names
        channels_to_exclude = config['component_estimation']["channels_to_exclude"]
        freq_limits = [config['component_estimation']['low_freq'],
                       config['component_estimation']['high_freq']]
        epoch_definition = config['export_clean']['epoch_definition']
        crop_seconds = [config['export_clean']['crop_seconds']]


        # Load the recording
        raw = aimind_mne.prepare_raw(
            config,
            bids_path,
            preload=True,
            channels_to_include=channels_to_include,
            channels_to_exclude=channels_to_exclude,
            freq_limits=freq_limits,
            crop_seconds=crop_seconds,
            badchannels_to_metadata=True,
            exclude_badchannels=True,
            set_annotations=True,
            epoch=epoch_definition)

        # If there is EOG or EKG, remove those components
        components_to_exclude = []
        if 'eog' in sobi.labels_.keys():
            components_to_exclude.append(sobi.labels_['eog'])
        if 'ecg' in sobi.labels_.keys():
            components_to_exclude.append(sobi.labels_['ecg'])
        components_to_exclude = sum(components_to_exclude, [])

        # If desired components, apply them.
        if len(components_to_exclude) > 0:
            # Remove the eog components
            sobi.apply(raw, exclude=components_to_exclude)


        # Export
        export_mne_epochs(raw,str(out_filepath))