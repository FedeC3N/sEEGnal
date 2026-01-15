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
    resample_frequency = config['component_estimation']['resampled_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]
    epoch_definition = config['artifact_detection']['EOG']['epoch_definition']

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

    # Extra outputs
    sfreq = raw.info['sfreq']
    last_sample = raw.last_samp

    # Apply SOBI and filters, and epoch the data
    raw = mne_tools.prepare_eeg(
        config,
        bids_path,
        raw=raw,
        preload=True,
        apply_sobi=sobi,
        freq_limits=freq_limits,
        epoch=epoch_definition
    )

    # Divide between frontal channels and background channels
    # Frontal
    frontal_channels = config['artifact_detection']['frontal_channels']
    frontal_channels = [
        current_channel for current_channel in frontal_channels if current_channel in channels_to_include
    ]
    if len(frontal_channels) > 0:
        frontal_raw = raw.copy()
        frontal_raw.pick(frontal_channels)
    else:
        return [], [], []  # If no frontal channels, no EOG artifacts

    # Background
    background_channels = [
        current_channel for current_channel in raw.ch_names if current_channel not in frontal_channels
    ]
    background_raw = raw.copy()
    background_raw.pick(background_channels)

    # Get the std
    frontal_raw_std = np.std(frontal_raw.get_data(),axis=2)
    background_raw_std = np.std(background_raw.get_data(), axis=2)

    # Get the median and MAD
    median = np.median(background_raw_std)
    mad = median_abs_deviation(background_raw_std,axis=None)

    # Look for significant activity
    hits = (frontal_raw_std - median) / mad > config[
        'artifact_detection']['EOG']['threshold']
    hits = np.sum(hits,axis=1) > 0

    # Save the EOG index
    EOG_index = raw.events[hits,0]

    # If you have crop the recordings, put the indexes according to the original number of samples
    if crop_seconds:
        crop_samples = crop_seconds * sfreq
        last_sample = int(last_sample + 2 * crop_samples)
        EOG_index = [int(current_index - crop_samples) for current_index in EOG_index]

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
    crop_seconds = config['component_estimation']['crop_seconds']
    resample_frequency = config['component_estimation']['resampled_frequency']
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
        metadata_badchannels=True,
        interpolate_badchannels=True,
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

    # Find the peaks of each component
    muscle_index = []
    for icomponent in range(muscle_components_time_courses.shape[0]):

        # Select the absolute current component
        current_component = muscle_components_time_courses[icomponent, :]
        current_component = np.abs(current_component)

        # Find peaks
        prominence = 10 * np.std(current_component)
        current_peaks, _ = find_peaks(
            current_component,
            prominence=prominence
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
        'desc': 'sobi_artifacts',
        'components_to_include': [],
        'components_to_exclude': ['eog', 'ecg']
    }
    freq_limits         = [
        config['artifact_detection']['sensor']['low_freq'],
        config['artifact_detection']['sensor']['high_freq']
    ]
    crop_seconds = config['component_estimation']['crop_seconds']
    resample_frequency = config['component_estimation']['resampled_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]
    epoch_definition = config['artifact_detection']['sensor']['epoch_definition']

    # Load the raw data
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

    # Extra outputs
    sfreq = raw.info['sfreq']
    last_sample = raw.last_samp

    # Apply SOBI and filters
    raw = mne_tools.prepare_eeg(
        config,
        bids_path,
        raw=raw,
        preload=True,
        apply_sobi=sobi,
        freq_limits=freq_limits,
        epoch=epoch_definition
    )

    # Discard first the muscle artefacts
    raw.drop_bad()

    # Get the clean data
    raw_data = raw.get_data().copy()

    # Estimate the std of each epoch
    raw_data_std = raw_data.std(axis=2)

    # Estimate the median and MAD
    median = np.median(raw_data_std)
    MAD = median_abs_deviation(raw_data_std,axis=None)

    # Find important deviations
    hits = (raw_data_std - median) / MAD > config['artifact_detection']['sensor']['threshold']
    hits = np.sum(hits,axis=1) > 0
    hits = np.where(hits)[0]

    # Save the peaks index
    sensor_index = raw.events[hits,:][:,0]

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
    crop_seconds = config['component_estimation']['crop_seconds']
    resample_frequency = config['component_estimation']['resampled_frequency']
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
        apply_sobi=sobi,
        freq_limits=freq_limits
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
