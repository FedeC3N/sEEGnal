# -*- coding: utf-8 -*-
"""

Find different types of badchannels.

EEG: high_impedance_badchannels, high_variance_badchannels, power_spectrum_badchannels

Created on March 26 2024
@author: Federico Ramirez
"""

# Imports
import scipy
import numpy as np

import sEEGnal.tools.bids_tools as bids
import sEEGnal.tools.mne_tools as aimind_mne


# Modules
def high_impedance_detection(config, bids_path):
    """

    Look for badchannels based on their impedance

    :arg
    config (dict): Configuration parameters (paths, parameters, etc)
    bids_path (dict): Path to the recording

    :returns
    List of badchannels

    """

    # Reads the channels information.
    channels = bids.read_chan(bids_path)

    # Identifies the channels with high impedance (higher than 200).
    hits = channels['impedance'] > config['badchannel_detection']['high_impedance']['threshold']
    high_impedance_badchannels = list(channels.loc[hits, 'name'])

    # Remove the excluded channels
    channels_to_exclude = config['global']['channels_to_exclude']
    high_impedance_badchannels = [
        current_channel for current_channel in high_impedance_badchannels if current_channel not in channels_to_exclude
    ]

    return high_impedance_badchannels


def impossible_amplitude_detection(config, bids_path, badchannels):
    """

    Look for channels with low amplitude

    :arg
    config (dict): Configuration parameters (paths, parameters, etc)
    bids_path (dict): Path to the recording

    :returns
    List of badchannels

    """

    # Parameters for loading EEG  recordings
    freq_limits = [
        config['badchannel_detection']['impossible_amplitude']['low_freq'],
        config['badchannel_detection']['impossible_amplitude']['high_freq']
    ]
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]
    epoch_definition = config['badchannel_detection']['impossible_amplitude']["epoch_definition"]

    # Load the raw data
    raw = aimind_mne.prepare_eeg(
        config,
        bids_path,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        freq_limits=freq_limits,
        exclude_badchannels=True,
        set_annotations=False,
        epoch=epoch_definition,
        rereference=False
    )

    # De-mean the channels
    raw_data = raw.get_data().copy()

    # Estimate the average standard deviation of each epoch
    raw_data_std = raw_data.std(axis=2)

    # Use the low and high threshold to define impossible amplitudes
    hits_low = raw_data_std < config['badchannel_detection']['impossible_amplitude']["low_threshold"]
    hits_high = raw_data_std > config['badchannel_detection']['impossible_amplitude']["high_threshold"]

    # Get the number of occurrences per channel in percentage
    hits = hits_low | hits_high
    hits = hits.sum(axis=0) / raw_data_std.shape[0]

    # Define as badchannel if many epochs are bads
    hits = np.flatnonzero(hits > config['badchannel_detection']['impossible_amplitude']["percentage_threshold"])
    impossible_amplitude_badchannels = [raw.ch_names[hit] for hit in hits]

    return impossible_amplitude_badchannels


def power_spectrum_detection(config, bids_path, badchannels):
    """

    Look for badchannels based on anomalies in the power spectrum to detect badchannels

    :arg
    config (dict): Configuration parameters (paths, parameters, etc)
    bids_path (dict): Path to the recording

    :returns
    List of badchannels

    """

    # Parameters for loading EEG  recordings
    freq_limits = [
        config['badchannel_detection']['pow_spectrum']['low_freq'],
        config['badchannel_detection']['pow_spectrum']['high_freq']
    ]
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]
    crop_seconds = [config['badchannel_detection']["crop_seconds"]]
    epoch_definition = config['badchannel_detection']['pow_spectrum']['epoch_definition']

    # Load the raw data
    raw = aimind_mne.prepare_eeg(
        config,
        bids_path,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        freq_limits=freq_limits,
        crop_seconds=crop_seconds,
        exclude_badchannels=True,
        set_annotations=False,
        epoch=epoch_definition,
        rereference=False
    )

    # Compute the power spectrum
    psd = raw.compute_psd(method='welch', fmin=freq_limits[0], fmax=freq_limits[1], average='mean')
    psd_data = psd.get_data().copy()

    # Compute the variance (equivalent to power)
    raw_data = raw.get_data().copy()
    var_data = raw_data.var(axis=2)

    # Get the average of each epoch and channel
    hits = np.empty((var_data.shape[0], var_data.shape[1]))
    for ichannel in range(var_data.shape[1]):

        # Remove the current channel to estimate the average value of the power spectrum in the rest of the recording
        current_channel_pow = var_data[:, ichannel]
        rest_pow = np.delete(var_data, ichannel, 1)
        rest_pow = rest_pow.sum(axis=1).mean(axis=0)

        # Find the epochs where the current channel is above the threshold
        threshold = config['badchannel_detection']['pow_spectrum']['threshold'] * rest_pow
        hits[:, ichannel] = current_channel_pow > threshold

    # Get the number of occurrences per channel in percentage
    hits = hits.sum(axis=0) / psd_data.shape[0]

    # Define as badchannel if many epochs are bads
    hits = np.flatnonzero(hits > config['badchannel_detection']['pow_spectrum']["percentage_threshold"])
    power_spectrum_badchannels = [raw.ch_names[hit] for hit in hits]

    return power_spectrum_badchannels


def gel_bridge_detection(config, bids_path, badchannels):
    """

    Look for badchannels based on gel bridges. It is based on correlation and the idea that the correlation must happen
    continuously in time over certain point.

    :arg
    config (dict): Configuration parameters (paths, parameters, etc)
    bids_path (dict): Path to the recording

    :returns
    List of badchannels

    """

    # Create empty list to append badchannels
    gel_bridge_badchannels = []

    # Read the ICA information
    sobi = bids.read_sobi(bids_path, 'sobi-badchannels')

    # Parameters for loading EEG  recordings
    freq_limits = [config['component_estimation']['low_freq'], config['component_estimation']['high_freq']]
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]
    crop_seconds = [config['badchannel_detection']["crop_seconds"]]

    # Load the raw EEG
    raw = aimind_mne.prepare_eeg(
        config,
        bids_path,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        freq_limits=freq_limits,
        crop_seconds=crop_seconds,
        exclude_badchannels=True,
        set_annotations=False,
        rereference=False
    )

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

    # Filter again in the desired frequencies
    freq_limits = [
        config['badchannel_detection']['gel_bridge']['low_freq'],
        config['badchannel_detection']['gel_bridge']['high_freq']
    ]
    raw.filter(freq_limits[0], freq_limits[1])

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


def high_deviation_detection(config, bids_path, badchannels):
    """

    Look for badchannels based on the channel variance

    :arg
    config (dict): Configuration parameters (paths, parameters, etc)
    bids_path (dict): Path to the recording

    :returns
    List of badchannels

    """

    # Read the ICA information
    sobi = bids.read_sobi(bids_path, 'sobi-badchannels')

    # Parameters for loading EEG  recordings
    freq_limits = [config['component_estimation']['low_freq'], config['component_estimation']['high_freq']]
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]
    crop_seconds = [config['badchannel_detection']["crop_seconds"]]
    epoch_definition = config['badchannel_detection']['high_deviation']['epoch_definition']

    # Load the raw data
    raw = aimind_mne.prepare_eeg(
        config,
        bids_path,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        freq_limits=freq_limits,
        crop_seconds=crop_seconds,
        exclude_badchannels=True,
        set_annotations=False,
        epoch=epoch_definition,
        rereference=False
    )

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

    # Filter again in the desired frequencies
    freq_limits = [
        config['badchannel_detection']['high_deviation']['low_freq'],
        config['badchannel_detection']['high_deviation']['high_freq']
    ]
    raw.filter(freq_limits[0], freq_limits[1])

    # Get a new copy of the data
    raw_data = raw.get_data().copy()

    # Get the average of each epoch and channel
    hits = np.empty((raw_data.shape[0], raw_data.shape[1]))
    for ichannel in range(len(raw.ch_names)):

        # Remove the current channel to estimate the average value of the power spectrum in the rest of the recording
        current_channel_std = raw_data[:, ichannel, :].std(axis=1)
        rest_std = np.delete(raw_data, ichannel, 1)
        rest_std = rest_std.std(axis=2).mean(axis=1)

        # Find the epochs where the current channel is above the threshold
        threshold = config['badchannel_detection']['high_deviation']['threshold'] * rest_std
        hits[:, ichannel] = current_channel_std > threshold

    # Get the number of occurrences per channel in percentage
    hits = hits.sum(axis=0) / hits.shape[0]

    # Define as badchannel if many epochs are bads
    hits = np.flatnonzero(hits > config['badchannel_detection']['high_deviation']["percentage_threshold"])
    high_deviation_badchannels = [raw.ch_names[hit] for hit in hits]

    return high_deviation_badchannels
