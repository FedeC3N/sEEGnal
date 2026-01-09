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

import sEEGnal.tools.bids_tools as bids
import sEEGnal.tools.mne_tools as mne_tools


# Modules
def EOG_detection(config, bids_path, frontal_channels='all'):
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
    crop_seconds        = config['artifact_detection']['EOG']['crop_seconds']
    resample_frequency  = config['artifact_detection']['EOG']['resampled_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]

    # Load the raw data
    raw = mne_tools.prepare_eeg(
        config,
        bids_path,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        apply_sobi=sobi,
        freq_limits=freq_limits,
        crop_seconds=crop_seconds,
        resample_frequency=resample_frequency,
        exclude_badchannels=True,
        interpolate_bads=True,
        rereference='average'
    )

    # Select no-frontal channels used to estimate the non-EOG deviation
    background_channels = [
        current_channel for current_channel in raw.ch_names if current_channel not in frontal_channels
    ]

    # Select the background data
    background_raw = raw.copy()
    background_raw.pick(background_channels)

    # Average deviation of the recording
    background_raw_data = background_raw.get_data().copy()
    background_average_deviation = background_raw_data.std(axis=1).mean()

    # Select EOG channels
    frontal_channels = [
        current_channel for current_channel in frontal_channels if current_channel in channels_to_include
    ]

    # Select the present frontal channels
    if len(frontal_channels) > 0:
        raw.pick(frontal_channels)

    # If no frontal channels, no EOG artifacts
    else:
        return [], [], []

    # Get the data
    channel_data = raw.get_data().copy()

    # Go one by one through the channels looking for EOG artifacts
    for ichannel in range(len(raw.ch_names)):

        current_channel_data = channel_data[ichannel, :] - channel_data[ichannel, :].mean()

        # Consider a peak everything above 10 std
        current_peaks = np.nonzero(
            abs(current_channel_data) > config['artifact_detection']['EOG']['ratio'] * background_average_deviation
        )[0]

        # Find the peaks
        # If it is the first channel, create the variable
        if ichannel == 0:
            EOG_index = current_peaks

        # If already calculated, append
        else:

            # If not empty, append
            if len(current_peaks) > 0:
                EOG_index = np.append(EOG_index, current_peaks)

    # Extra outputs
    last_sample = raw.last_samp
    sfreq = raw.info['sfreq']

    # If you have crop the recordings, put the indexes according to the original number of samples
    if crop_seconds:
        crop_samples = crop_seconds * sfreq
        last_sample = int(last_sample + 2 * crop_samples)
        EOG_index = [int(current_index + crop_samples) for current_index in EOG_index]

    return EOG_index, last_sample, sfreq


def muscle_detection(config, bids_path):
    """

    Detect muscle artifacts

    :arg
    config (dict): Configuration parameters (paths, parameters, etc)
    bids_path (dict): Path to the recording
    channels_to_include (str): Channels type to work on.

    :returns
    MNE annotation object with detected muscle artifacts

    """

    # Read the ICA information
    sobi = bids.read_sobi(bids_path, 'sobi_artifacts')

    # Parameters for loading EEG recordings
    freq_limits         = [
        config['component_estimation']['low_freq'],
        config['component_estimation']['high_freq']
    ]
    crop_seconds        = config['artifact_detection']['muscle']['crop_seconds']
    resample_frequency  = config['artifact_detection']['muscle']['resampled_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]

    # Load raw EEG
    raw = mne_tools.prepare_eeg(
        config,
        bids_path,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        freq_limits=freq_limits,
        crop_seconds=crop_seconds,
        resample_frequency=resample_frequency,
        exclude_badchannels=True,
        interpolate_bads=True,
        rereference='average'
    )

    # Get the muscle components time series
    if len(sobi.labels_['muscle']) > 0:

        # Get the sources of interest
        sources = sobi.get_sources(raw)
        sources.pick(sobi.labels_['muscle'])

        muscle_components_time_courses = sources.get_data().copy()

    # If no muscular, we use the whole matrix but without eog (it confused the muscle artifact detection)
    elif len(sobi.labels_['eog']) > 0:

        # Get the sources of interest
        ic_labels_without_EOG = sobi.labels_.copy()
        ic_labels_without_EOG.pop('eog')
        ic_labels_without_EOG = [i for i in ic_labels_without_EOG.values()]
        ic_labels_without_EOG = sum(ic_labels_without_EOG, [])
        sources = sobi.get_sources(raw)
        sources.pick(ic_labels_without_EOG)

        # Get the data
        muscle_components_time_courses = sources.get_data().copy()

    # If not, used the whole matrix
    else:

        # Now, the "muscle components" are all the components
        sources = sobi.get_sources(raw)
        muscle_components_time_courses = sources.get_data().copy()

    # Find the peaks of each channel (peak = signal > 10*std)
    muscle_components_time_courses_std = muscle_components_time_courses.std(axis=1)
    muscle_index = []
    for ichannel in range(len(muscle_components_time_courses_std)):

        current_channel = muscle_components_time_courses[ichannel, :]
        current_std = muscle_components_time_courses_std[ichannel]
        current_peaks, _ = find_peaks(
            abs(current_channel), height=config['artifact_detection']['muscle']['threshold'] * current_std
        )

        # If any, add to list
        if len(current_peaks) > 0:
            muscle_index = muscle_index + current_peaks.tolist()

    # Extra outputs
    last_sample = raw.last_samp
    sfreq = raw.info['sfreq']

    # If you have crop the recordings, put the indexes according to the original number of samples
    if crop_seconds:
        crop_samples = crop_seconds * sfreq
        last_sample = int(last_sample + 2 * crop_samples)
        muscle_index = [int(current_index + crop_samples) for current_index in muscle_index]

    return muscle_index, last_sample, sfreq


def sensor_detection(config, bids_path):
    """

    Detect sensor artifacts (jumps)

    :arg
    config (dict): Configuration parameters (paths, parameters, etc)
    bids_path (dict): Path to the recording
    channels_to_include (str): Channels type to work on.

    :returns
    MNE annotation object with detected sensor artifacts

    """

    # Parameters for loading EEG recordings
    sobi                = {
        'desc': 'sobi',
        'components_to_include': [],
        'components_to_exclude': ['eog', 'ecg']
    }
    freq_limits         = [
        config['artifact_detection']['sensor']['low_freq'],
        config['artifact_detection']['sensor']['high_freq']
    ]
    crop_seconds        = config['artifact_detection']['sensor']['crop_seconds']
    resample_frequency  = config['artifact_detection']['sensor']['resampled_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]

    # Load the raw data
    raw = mne_tools.prepare_eeg(
        config,
        bids_path,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        apply_sobi=sobi,
        crop_seconds=crop_seconds,
        freq_limits=freq_limits,
        resample_frequency=resample_frequency,
        exclude_badchannels=True,
        interpolate_bads=True,
        set_annotations=True,
        rereference='average'
    )

    # Create Epoch object of 1 second to estimate the average std discarding first the muscle artefacts
    dummy_events = mne.make_fixed_length_events(raw, duration=0.5)
    raw_epoched = mne.Epochs(raw, events=dummy_events, reject_by_annotation=True, baseline=None, tmin=0, tmax=0.5)
    raw_epoched.drop_bad()

    # Get the clean data
    raw_data = raw_epoched.get_data().copy()

    # Estimate the std of each channel
    raw_data_std = raw_data.std(axis=2)
    raw_data_std_average = raw_data_std.mean(axis=0)

    # Get the original data
    raw_data = raw.get_data().copy()

    # Find the peaks of each channel (peak = signal > 3*sensor_threshold)
    # Empty list to save the peaks
    sensor_index = []

    # Windows size
    window_size = int(np.floor(0.1 * raw.info['sfreq']))

    # Use a sliding windows to find jumps. A jump is defined as a windows with 3*std
    for i in range(raw_data.shape[1] - window_size + 1):

        # Get the data of the current window
        current_window = (raw_data[:, i:i + window_size])

        # Check if the std of any channel is > 3*raw_std
        if any(current_window.std(axis=1) > config['artifact_detection']['sensor']['ratio'] * raw_data_std_average):
            sensor_index = sensor_index + np.arange(i, i + window_size).tolist()
            sensor_index = list(set(sensor_index))

    # Extra outputs
    last_sample = raw.last_samp
    sfreq = raw.info['sfreq']

    # If you have crop the recordings, put the indexes according to the original number of samples
    if crop_seconds:
        crop_samples = crop_seconds * sfreq
        last_sample = int(last_sample + 2 * crop_samples)
        sensor_index = [int(current_index + crop_samples) for current_index in sensor_index]

    return sensor_index, last_sample, sfreq


def other_detection(config, bids_path):
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
        'desc': 'sobi',
        'components_to_include': [],
        'components_to_exclude': ['eog', 'ecg']
    }
    freq_limits         = [
        config['artifact_detection']['other']['low_freq'],
        config['artifact_detection']['other']['high_freq']
    ]
    crop_seconds        = config['artifact_detection']['other']['crop_seconds']
    resample_frequency  = config['artifact_detection']['other']['resampled_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]

    # Load the raw data
    raw = mne_tools.prepare_eeg(
        config,
        bids_path,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        apply_sobi=sobi,
        resample_frequency=resample_frequency,
        freq_limits=freq_limits,
        crop_seconds=crop_seconds,
        exclude_badchannels=True,
        interpolate_bads=True,
        set_annotations=True,
        rereference='average'
    )

    # De-mean the channels
    raw_data = raw.get_data().copy()

    # Estimate the average standard deviation of each epoch
    raw_data_demean_abs = np.abs(raw_data)

    # Check if the std of any channel is > 3*raw_std
    other_index = []
    for ichannel in range(raw_data_demean_abs.shape[0]):

        current_channel = raw_data_demean_abs[ichannel, :]
        current_peaks, _ = find_peaks(current_channel, height=config['artifact_detection']['other']['threshold'])

        # If any, add to list
        if len(current_peaks) > 0:
            other_index = other_index + current_peaks.tolist()

    # Extra outputs
    last_sample = raw.last_samp
    sfreq = raw.info['sfreq']

    # If you have crop the recordings, put the indexes according to the original number of samples
    if crop_seconds:
        crop_samples = crop_seconds * sfreq
        last_sample = int(last_sample + 2 * crop_samples)
        other_index = [int(current_index + crop_samples) for current_index in other_index]

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
