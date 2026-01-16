# -*- coding: utf-8 -*-
"""

Find different types of badchannels.

EEG: high_impedance_badchannels, high_variance_badchannels, power_spectrum_badchannels

Created on March 26 2024
@author: Federico Ramirez
"""

# Imports
import numpy as np
from scipy.stats import median_abs_deviation

import sEEGnal.tools.mne_tools as mne_tools


# Modules
def impossible_amplitude_detection(config, bids_path):
    """

    Look for channels with low amplitude

    :arg
    config (dict): Configuration parameters (paths, parameters, etc)
    bids_path (dict): Path to the recording

    :returns
    List of badchannels

    """

    # Parameters for loading EEG  recordings
    channels_to_include = config['global']['channels_to_include']
    channels_to_exclude = config['global']['channels_to_exclude']
    freq_limits         = [
        config['badchannel_detection']['impossible_amplitude']['low_freq'],
        config['badchannel_detection']['impossible_amplitude']['high_freq']
    ]
    crop_seconds        = config['badchannel_detection']['crop_seconds']
    epoch_definition    = config['badchannel_detection']['impossible_amplitude']['epoch_definition']

    # Load the raw data
    raw = mne_tools.prepare_eeg(
        config,
        bids_path,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        notch_filter=True,
        freq_limits=freq_limits,
        crop_seconds=crop_seconds,
        metadata_badchannels=True,
        exclude_badchannels=True,
        epoch=epoch_definition,
        rereference=False
    )

    # De-mean the channels
    raw_data = raw.get_data().copy()

    # Estimate the average standard deviation of each epoch
    raw_data_std = raw_data.std(axis=2)

    # Use the low and high threshold to define impossible amplitudes
    hits_low = raw_data_std < config['badchannel_detection']['impossible_amplitude']['low_threshold']
    hits_high = raw_data_std > config['badchannel_detection']['impossible_amplitude']['high_threshold']

    # Get the number of occurrences per channel in percentage
    hits = hits_low | hits_high
    hits = hits.sum(axis=0) / raw_data_std.shape[0]

    # Define as badchannel if many epochs are bads
    hits = np.flatnonzero(hits > config['badchannel_detection']['impossible_amplitude']['percentage_threshold'])
    impossible_amplitude_badchannels = [raw.ch_names[hit] for hit in hits]

    return impossible_amplitude_badchannels


def power_spectrum_detection(config, bids_path):
    """

    Look for badchannels based on anomalies in the power spectrum to detect badchannels

    :arg
    config (dict): Configuration parameters (paths, parameters, etc)
    bids_path (dict): Path to the recording

    :returns
    List of badchannels

    """

    # Parameters for loading EEG  recordings
    freq_limits         = [
        config['badchannel_detection']['pow_spectrum']['low_freq'],
        config['badchannel_detection']['pow_spectrum']['high_freq']
    ]
    channels_to_include = config['global']['channels_to_include']
    channels_to_exclude = config['global']['channels_to_exclude']
    crop_seconds        = config['badchannel_detection']['crop_seconds']
    epoch_definition    = config['badchannel_detection']['pow_spectrum']['epoch_definition']

    # Load the raw data
    raw = mne_tools.prepare_eeg(
        config,
        bids_path,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        freq_limits=freq_limits,
        crop_seconds=crop_seconds,
        metadata_badchannels=True,
        exclude_badchannels=True,
        epoch=epoch_definition
    )

    # Compute the power spectrum
    psd             = raw.compute_psd(method='welch', fmin=freq_limits[0], fmax=freq_limits[1], average='mean')
    band_power      = np.trapezoid(psd.get_data(),psd.freqs,axis=2)
    band_power_log  = np.log10(band_power)

    # Estimate the median and the Median Absolute Deviation
    median = np.median(band_power_log)
    MAD = median_abs_deviation(band_power_log, axis=None)

    # Define bad channels using Z-score
    bad_epochs = ((band_power_log - median) / MAD > config['badchannel_detection']['pow_spectrum']['threshold'])

    # Get the number of occurrences per channel in percentage
    hits = bad_epochs.sum(axis=0) / bad_epochs.shape[0]

    # Define as badchannel if many epochs are bads
    hits = np.flatnonzero(hits > config['badchannel_detection']['pow_spectrum']['percentage_threshold'])
    power_spectrum_badchannels = [raw.ch_names[hit] for hit in hits]

    return power_spectrum_badchannels


def gel_bridge_detection(config, bids_path):
    """

    Look for badchannels based on gel bridges. It is based on correlation and physical distance

    :arg
    config (dict): Configuration parameters (paths, parameters, etc)
    bids_path (dict): Path to the recording

    :returns
    List of badchannels

    """

    # Create empty list to append badchannels
    gel_bridge_badchannels = []

    # Parameters for loading EEG  recordings
    sobi                = {
        'desc': 'sobi-badchannels',
        'components_to_include': [],
        'components_to_exclude': ['eog','ecg']
    }
    freq_limits         = [
        config['badchannel_detection']['gel_bridge']['low_freq'],
        config['badchannel_detection']['gel_bridge']['high_freq']
    ]
    resample_frequency  = config['component_estimation']['resampled_frequency']
    channels_to_include = config['global']['channels_to_include']
    channels_to_exclude = config['global']['channels_to_exclude']
    crop_seconds        = config['badchannel_detection']['crop_seconds']

    # Load the raw EEG
    raw = mne_tools.prepare_eeg(
        config,
        bids_path,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        resample_frequency=resample_frequency,
        crop_seconds=crop_seconds
    )

    # Apply SOBI
    raw = mne_tools.prepare_eeg(
        config,
        bids_path,
        raw=raw,
        apply_sobi=sobi,
    )

    # Filter
    raw = mne_tools.prepare_eeg(
        config,
        bids_path,
        raw=raw,
        freq_limits=freq_limits,
        notch_filter=True,
        metadata_badchannels=True,
        exclude_badchannels=True
    )

    # Get the data
    raw_data = raw.get_data().copy()

    # Estimate the correlation
    correlation_coefficients = np.corrcoef(raw_data)

    # Find highly correlated channels
    correlation_coefficients = np.triu(correlation_coefficients, k=1)
    correlation_coefficients_mask = correlation_coefficients > config['badchannel_detection']['gel_bridge']['threshold']

    # Find indexes of channels with sequences indicating gel-bridge.
    row_ind, col_ind = np.where(correlation_coefficients_mask)

    # If any gel_bridged badchannel, check if they are neighbours
    if len(row_ind) > 0:

        # Load the montage to assure they are neighbours (5cm for example).
        montage = raw.get_montage()
        channels_position = montage.get_positions()
        channels_position = channels_position['ch_pos']
        channels_position = channels_position.values()
        channels_position = list(channels_position)
        channels_position = np.array(channels_position)

        for ichannel in range(len(row_ind)):

            # Check the distance to assure they are neighbours (5cm for example).
            ch_pos1 = channels_position[row_ind[ichannel], :]
            ch_pos2 = channels_position[col_ind[ichannel], :]
            distance = np.linalg.norm(ch_pos1 - ch_pos2)

            # If the channels are close enough, they are gel-bridged
            if distance < config['badchannel_detection']['gel_bridge']['neighbour_distance']:

                gel_bridge_badchannels.append(montage.ch_names[row_ind[ichannel]])
                gel_bridge_badchannels.append(montage.ch_names[col_ind[ichannel]])

    return gel_bridge_badchannels


def high_deviation_detection(config, bids_path):
    """

    Look for badchannels based on the channel variance

    :arg
    config (dict): Configuration parameters (paths, parameters, etc)
    bids_path (dict): Path to the recording

    :returns
    List of badchannels

    """

    # Parameters for loading EEG  recordings
    sobi = {
        'desc': 'sobi-badchannels',
        'components_to_include': [],
        'components_to_exclude': ['eog', 'ecg']
    }
    freq_limits = [
        config['badchannel_detection']['high_deviation']['low_freq'],
        config['badchannel_detection']['high_deviation']['high_freq']
    ]
    resample_frequency = config['component_estimation']['resampled_frequency']
    channels_to_include = config['global']['channels_to_include']
    channels_to_exclude = config['global']['channels_to_exclude']
    crop_seconds = config['badchannel_detection']['crop_seconds']
    epoch_definition = config['badchannel_detection']['high_deviation']['epoch_definition']

    # Load the raw EEG
    raw = mne_tools.prepare_eeg(
        config,
        bids_path,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        resample_frequency=resample_frequency,
        crop_seconds=crop_seconds
    )

    # Apply SOBI
    raw = mne_tools.prepare_eeg(
        config,
        bids_path,
        raw=raw,
        apply_sobi=sobi
    )

    # Filter
    raw = mne_tools.prepare_eeg(
        config,
        bids_path,
        raw=raw,
        preload=True,
        freq_limits=freq_limits,
        notch_filter=True,
        metadata_badchannels=True,
        exclude_badchannels=True,
        epoch=epoch_definition
    )

    # Get a new copy of the data and estimate std
    raw_data = raw.get_data().copy()
    raw_data_std = np.std(raw_data,axis=2)

    # Estimate the median and the Median Absolute Deviation
    median = np.median(raw_data_std)
    mad = median_abs_deviation(raw_data_std,axis=None)

    # Define bad channels using Z-score
    hits = ((raw_data_std - median) / mad >
                       config['badchannel_detection']['high_deviation'][
                           'threshold'])

    # Get the number of occurrences per channel in percentage
    hits = hits.sum(axis=0) / hits.shape[0]

    # Define as badchannel if many epochs are bads
    hits = np.flatnonzero(hits > config['badchannel_detection']['high_deviation']['percentage_threshold'])
    high_deviation_badchannels = [raw.ch_names[hit] for hit in hits]

    return high_deviation_badchannels
