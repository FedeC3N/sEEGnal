# -*- coding: utf-8 -*-
"""

Convert the raw data structure to BIDS format

Created on Mon May 1 11:30:42 2023
@author: Fede

"""

# Imports
import os
import traceback
from datetime import datetime as dt, timezone

import mne
import numpy
import mne_bids

import sEEGnal.io.eep as eep
from sEEGnal.tools.measure_performance import measure_performance
from sEEGnal.io.read_source_files import read_source_files



# Set the output levels
mne.utils.set_log_level(verbose='ERROR')



# Modules
@measure_performance
def standardize(config,current_file,bids_path):
    """

    Checks the type of file to standardize

    :arg
    config (dict): Configuration parameters (paths, parameters, etc)
    current_file (str): Name of the file to process
    bids_path (BIDSpath): Associated bids_path

    :returns
    A dict with the result of the process

    """

    # For EEG
    if bids_path.datatype == 'eeg':

        try:

            # Standardize
            bids_basename = standardize_eeg_file(config,current_file, bids_path)

            # Save the results
            now = dt.now(timezone.utc)
            formatted_now = now.strftime("%d-%m-%Y %H:%M:%S")
            results = {'result': 'ok',
                       'file': current_file,
                       'bids_basename': bids_basename,
                       "date":formatted_now
                       }

        except Exception as e:

            # Save the error
            now = dt.now(timezone.utc)
            formatted_now = now.strftime("%d-%m-%Y %H:%M:%S")
            results = {'result': 'error',
                       'file': current_file,
                       'details': f"Exception: {str(e)}, {traceback.format_exc()}",
                       'date': formatted_now
                       }

    else:

        # Not accepted type to process
        now = dt.now(timezone.utc)
        formatted_now = now.strftime("%d-%m-%Y %H:%M:%S")
        results = {'result':'error',
                   'file':current_file,
                   'details': 'Not accepted type of file to process',
                   'date': formatted_now
                   }

    return results



def standardize_eeg_file(config, current_file, bids_path):
    """

    Standardize the EEG recordings to BIDS format.

    :arg
    config (dict): Configuration parameters (paths, parameters, etc)
    bids_path (BIDSpath): Metadata of the bids_path to process

    :returns
    A string with the BIDS path of the recording processed.

    """

    # Reads the data as an MNE object.
    source_filepath = os.path.join(config['path']['sourcedata'],current_file)
    mnedata = read_source_files(config,source_filepath)

    # Adds some project-specific information.
    mnedata.info['line_freq'] = 50
    mnedata.info['subject_info'] = dict()
    mnedata.info['subject_info']['his_id'] = bids_path.subject

    # Writes the data into the BIDS path.
    mne_bids.write_raw_bids(
        mnedata, bids_path,
        allow_preload=True,
        format='BrainVision',
        overwrite=True)

    # Reads the channels information.
    ch_tsv = bids_path.copy().update(suffix='channels', extension='.tsv')
    ch_data = mne_bids.tsv_handler._from_tsv(ch_tsv)

    # Sets the anti-alias filter to 26% of the sampling rate.
    aaf = 0.26 * mnedata.info['sfreq']
    ch_aaf = ch_data['high_cutoff']
    ch_aaf = [aaf for ch in ch_aaf]
    ch_data['high_cutoff'] = ch_aaf

    # Add impedances to the channels file.
    ch_imp = mne_bids.dig._get_impedances(mnedata, ch_data['name'])
    if ch_imp[0] != 'n/a':
        ch_imp = [numpy.round(imp, 2) for imp in ch_imp]
    ch_data['impedance'] = ch_imp

    # Writes the updated channel information.
    mne_bids.tsv_handler._to_tsv(ch_data, ch_tsv)

    return bids_path.basename







