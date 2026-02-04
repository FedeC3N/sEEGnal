# -*- coding: utf-8 -*-
"""

Find different types of artifacts: EOG, muscle, sensor.

Created on June 19 2023
@author: Federico Ramirez
"""

# Imports
import mne
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation

import sEEGnal.tools.bids_tools as bids
import sEEGnal.tools.mne_tools as mne_tools


# Modules
def EOG_detection(config, bids_path):
    """

    Detect EOG artifacts

    :arg
    config (dict): Configuration parameters (paths, parameters, etc)
    raw (MNE.raw): Recording to work on.

    :returns
    Index where EOG where detected

    """

    # Parameters for loading EEG recordings
    sobi                = {
        'desc': 'sobi',
        'components_to_include': [],
        'components_to_exclude': ['eog','ecg']
    }
    freq_limits         = [
        config['artifact_detection']['EOG']['low_freq'],
        config['artifact_detection']['EOG']['high_freq']
    ]
    crop_seconds = config['component_estimation']['crop_seconds']
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]

    # Load the raw and apply SOBI
    raw = mne_tools.prepare_eeg(
        config,
        bids_path,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        notch_filter=True,
        resample_frequency=resample_frequency,
        metadata_badchannels=True,
        interpolate_badchannels=True,
        set_annotations=True,
        crop_seconds=crop_seconds,
        rereference='average'
    )

    # Apply SOBI and filters, and epoch the data
    raw = mne_tools.prepare_eeg(
        config,
        bids_path,
        raw=raw,
        preload=True,
        apply_sobi=sobi,
        freq_limits=freq_limits
    )

    # Select frontal channels
    frontal_channels = config['artifact_detection']['frontal_channels']
    frontal_channels = [
        current_channel for current_channel in frontal_channels if current_channel in channels_to_include
    ]
    if len(frontal_channels) > 0:
        frontal_raw = raw.copy()
        frontal_raw.pick(frontal_channels)
    else:
        return [], [], []  # If no frontal channels, no EOG artifacts

    # Find peaks after averaging the frontal channels
    frontal_data = np.mean(frontal_raw.get_data(),axis=0)
    hits = (np.abs(frontal_data) >
            config['artifact_detection']['EOG']['threshold'] * np.std(frontal_data))
    EOG_index = np.where(hits)[0]

    # Extra outputs
    last_sample = raw.original_last_samp
    sfreq = raw.info['sfreq']

    # If you have crop the recordings, put the indexes according to the original number of samples
    if crop_seconds:
        crop_samples = crop_seconds * sfreq
        EOG_index = [int(current_index + crop_samples) for current_index in
                        EOG_index]

    return EOG_index, last_sample, sfreq


def muscle_detection(config, bids_path, derivatives_label):
    """

    Detect muscle artifacts

    :arg
    config (dict): Configuration parameters (paths, parameters, etc)
    bids_path (dict): Path to the recording
    derivatives_label (str): Select the correct SOBI

    :returns
    MNE annotation object with detected muscle artifacts

    """

    # Parameters for loading EEG recordings
    freq_limits = [
        config['artifact_detection']['muscle']['low_freq'],
        config['artifact_detection']['muscle']['high_freq']
    ]
    crop_seconds = config['component_estimation']['crop_seconds']
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]

    # Load the raw and apply SOBI
    raw = mne_tools.prepare_eeg(
        config,
        bids_path,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        resample_frequency=resample_frequency,
        notch_filter=True,
        freq_limits=freq_limits,
        crop_seconds=crop_seconds,
        metadata_badchannels=True,
        exclude_badchannels=True
    )

    # Estimate the std across channels
    raw_std = np.std(raw.get_data(),axis=0)

    # Find peaks based on the total height (demeaning the signal first)
    raw_std = raw_std - np.mean(raw_std)
    height = (config['artifact_detection']['muscle']['threshold']
              * np.std(raw_std))
    muscle_index, _ = find_peaks(
        raw_std,
        height=height
    )

    # Extra outputs
    last_sample = raw.original_last_samp
    sfreq = raw.info['sfreq']

    # If you have crop the recordings, put the indexes according to the original number of samples
    if crop_seconds:
        crop_samples = crop_seconds * sfreq
        muscle_index = [int(current_index + crop_samples) for current_index in muscle_index]

    return muscle_index, last_sample, sfreq


def sensor_detection(config, bids_path,derivatives_label):
    """

    Detect sensor artifacts (jumps)

    :arg
    config (dict): Configuration parameters (paths, parameters, etc)
    bids_path (dict): Path to the recording
    derivatives_label (str): Select the correct SOBI

    :returns
    MNE annotation object with detected sensor artifacts

    """

    # Parameters for loading EEG recordings
    sobi                = {
        'desc': derivatives_label,
        'components_to_include': [],
        'components_to_exclude': ['eog', 'ecg']
    }
    freq_limits         = [
        config['artifact_detection']['sensor']['low_freq'],
        config['artifact_detection']['sensor']['high_freq']
    ]
    crop_seconds = config['component_estimation']['crop_seconds']
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]
    epoch_definition = config['artifact_detection']['sensor']['epoch_definition']
    epoch_definition['overlap'] = 0

    # Load the raw data
    raw = mne_tools.prepare_eeg(
        config,
        bids_path,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        notch_filter=True,
        resample_frequency=resample_frequency,
        metadata_badchannels=True,
        interpolate_badchannels=True,
        set_annotations=True,
        crop_seconds=crop_seconds,
        rereference='average'
    )

    # Apply SOBI and filters
    raw = mne_tools.prepare_eeg(
        config,
        bids_path,
        raw=raw,
        preload=True,
        freq_limits=freq_limits,
        epoch_definition=epoch_definition,
        apply_sobi=sobi
    )

    # Get the clean data
    raw_data    = raw.get_data()

    # Get the std for each channel and epoch
    raw_data_std = np.std(raw_data, axis=2)
    raw_data_std = np.max(np.abs(raw_data), axis=2)

    # For each epoch, estimate the threshold based on the std of the channels
    # in that epoch
    threshold = np.mean(raw_data_std,axis=1) * config['artifact_detection']['sensor']['threshold']

    # Find peaks based on the threshold
    threshold = np.repeat(threshold[:, np.newaxis], raw_data_std.shape[1], axis=1)
    hits = np.any(raw_data_std > threshold, axis=1)
    hits = np.where(hits)[0]

    # Save the peaks index
    sensor_index = raw.events[hits, :][:, 0]

    # Extra outputs
    last_sample = raw.original_last_samp
    sfreq = raw.info['sfreq']

    return sensor_index, last_sample, sfreq


def other_detection(config, bids_path,derivatives_label):
    """

    Look for signal with impossible values

    :arg
    config (dict): Configuration parameters (paths, parameters, etc)
    bids_path (dict): Path to the recording

    :returns
    List of badchannels

    """

    # Parameters for loading EEG recordings
    sobi                = {
        'desc': derivatives_label,
        'components_to_include': [],
        'components_to_exclude': ['eog', 'ecg']
    }
    freq_limits         = [
        config['artifact_detection']['other']['low_freq'],
        config['artifact_detection']['other']['high_freq']
    ]
    crop_seconds = config['component_estimation']['crop_seconds']
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]
    epoch_definition = config['artifact_detection']['other'][
        'epoch_definition']

    # Load the raw and apply SOBI
    raw = mne_tools.prepare_eeg(
        config,
        bids_path,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        resample_frequency=resample_frequency,
        metadata_badchannels=True,
        interpolate_badchannels=True,
        set_annotations=True,
        crop_seconds=crop_seconds,
        rereference='average'
    )

    # Apply SOBI and filters
    raw = mne_tools.prepare_eeg(
        config,
        bids_path,
        raw=raw,
        preload=True,
        apply_sobi=sobi,
        freq_limits=freq_limits,
        epoch_definition=epoch_definition
    )

    # De-mean the channels
    raw_data = raw.get_data().copy()
    epoch_average = np.mean(raw_data,axis=2)
    raw_data_demean = raw_data - epoch_average[:, :, np.newaxis]

    # Remove the offset
    raw_data_demean_abs = np.abs(raw_data_demean)

    # Find impossible peaks
    other_index = raw_data_demean_abs > config['artifact_detection']['other']['threshold']
    other_index = np.sum(np.sum(other_index,axis=2),axis=1)
    other_index = np.where(other_index)[0]
    other_index = raw.events[other_index,0]

    # Extra outputs
    last_sample = raw.original_last_samp
    sfreq = raw.info['sfreq']

    return other_index, last_sample, sfreq


def create_annotations(peaks_index, last_sample, sfreq, annotation_description, fictional_artifact_duration=0.5):
    """

    Create artifacts with certain duration.

    :arg
    peaks_index (list): Samples where the peaks are.
    last_sample (int): Limit of the recording
    sfreq (int): Sample frequency
    annotation_description (str): Name to store the annotation.
    fictional_artifact_duration_in_samples (int): Define an arbitrary duration for the artifact

    :returns
    Annotations

    """

    # Transform duration into samples
    fictional_artifact_duration_in_samples = int(fictional_artifact_duration * sfreq)

    # Convert peaks into artifact
    new_peaks_index, duration_in_samples = merge_peaks(peaks_index, last_sample, fictional_artifact_duration_in_samples)

    # Create the MNE annotations
    new_seconds_start = new_peaks_index / sfreq
    artifact_duration = duration_in_samples / sfreq
    description = annotation_description
    annotations = mne.Annotations(new_seconds_start, artifact_duration, description)

    return annotations


def merge_peaks(peaks_index, last_sample, fictional_artifact_duration_in_samples):
    """

    Create artifacts with certain duration.
    If several peaks are found within the windows duration, they are included as one artifact.
    If some artifacts overlap, it merges them.

    :arg
    peaks_index (list): Samples where the peaks are.
    last_sample (int): Limit of the recording
    fictional_artifact_duration_in_samples (int): Define an arbitrary duration for the artifact

    :returns
    The onset sample and duration of the merged artifact

    """

    # Convert the peaks into a vector with zeros and ones
    peaks_vector = np.zeros([last_sample])
    peaks_vector[peaks_index] = 1

    # Initialize output
    new_peak_index = []

    # With a sliding windows, save the index where peaks are present within the current sliding window position
    for i in range(0, len(peaks_vector) - fictional_artifact_duration_in_samples + 1,
                   fictional_artifact_duration_in_samples):

        # Get the samples within the current window position
        if fictional_artifact_duration_in_samples < last_sample:
            current_values = peaks_vector[i + 1:i + 1 + fictional_artifact_duration_in_samples].copy()

        else:
            current_values = peaks_vector[i + 1:].copy()

        # If any peak, mark the onset sample of the windows as the new index of the artifact
        if np.any(current_values):
            new_peak_index = new_peak_index + [i]

    # Estimate the end of the artifacts using 'fictional_artifact_duration_in_samples'
    new_peak_index = np.array(new_peak_index)
    end_peak_index = [x + fictional_artifact_duration_in_samples for x in new_peak_index]
    end_peak_index = np.array(end_peak_index)

    # If any windows touch each other, merge them
    for iartifact in range(1, len(new_peak_index) - 1):
        if (end_peak_index[iartifact] - new_peak_index[iartifact + 1]) == 0:
            new_peak_index[iartifact + 1] = new_peak_index[iartifact]
            new_peak_index[iartifact] = -1000

    # Update the new merged peaks
    mask = new_peak_index > 0
    new_peak_index = new_peak_index[mask]
    end_peak_index = end_peak_index[mask]

    # Define the artifact duration
    duration_in_samples = end_peak_index - new_peak_index

    return new_peak_index, duration_in_samples
