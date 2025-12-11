# -*- coding: utf-8 -*-
"""
Decide how to read a original EEG file based on the file type

Created on Thu 21/11/2024

@author: Fede
"""

# Imports
import numpy as np
import mne
import sEEGnal.io.eep as eep


def read_source_files(source_filepath):

    # Detect the EEG file type based on the extension
    extension = source_filepath.split('.')[-1]

    # Brain Vision
    if extension == 'vhdr':

        # Read the file
        mnedata = mne.io.read_raw_brainvision(source_filepath, preload=True)

        # Create RawArray
        mnedata = mne.io.RawArray(mnedata.get_data(), mnedata.info)

    elif extension == 'cnt':

        # Read the data using Ricardo's function
        mnedata = eep.read_mne(source_filepath)

    elif extension == 'set':

        # Read the file
        dummy_mnedata = mne.io.read_epochs_eeglab(source_filepath)

        # Transform to Raw object
        pseudodata = dummy_mnedata.get_data()
        pseudodata = np.hstack(pseudodata)
        mnedata = mne.io.RawArray(pseudodata, dummy_mnedata.info)
        mnedata.drop_channels('VEOG')

        # Get the montage
        desired_montage = mne.channels.make_standard_montage('easycap-M10')
        ch_names = mnedata.ch_names
        desired_montage.ch_names = ch_names
        mnedata.set_montage(desired_montage)

    else:

        mnedata = []

    return mnedata
