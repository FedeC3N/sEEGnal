# -*- coding: utf-8 -*-
"""

This module looks for different EEG artifacts.
First, it estimates the SOBI components (after excluding badchannels) and detects muscle and jumps artifacts. These
artifacts tend to affect SOBI estimation of other components.
After marking muscle and jumps artifacts, it estimates again SOBI components taking the artifacts out.
Then looks for EOG artifacts.

Created on September 27 08:43 2023
@author: Federico Ramirez

"""

# Imports
import traceback
from datetime import datetime as dt, timezone

import mne
import mne_icalabel as iclabel

import sEEGnal.tools.bids_tools as bids
import sEEGnal.tools.mne_tools as mnetools
import sEEGnal.preprocess.find_artifacts as find_artifacts



# Set the output levels
mne.utils.set_log_level(verbose='ERROR')


# Modules
def artifact_detection(config, bids_path):
    """

    Checks if it is a EEG file and call the correspondent function for artifact detection.

    :arg
    config (dict): Configuration parameters (paths, parameters, etc)
    bids_path (BIDSpath): Metadata to process

    :returns
    A dict with the result of the process

    """

    # For EEG
    if bids_path.datatype == 'eeg':

        try:

            # Detect badchannels in the current recording
            annotations = eeg_artifact_detection(config,bids_path)

            # Save the results
            now = dt.now(timezone.utc)
            formatted_now = now.strftime("%d-%m-%Y %H:%M:%S")
            results = {'result':'ok',
                       'bids_basename':bids_path.basename,
                       "date":formatted_now,
                       'annotations':annotations
                       }

        except Exception as e:

            # Save the error
            now = dt.now(timezone.utc)
            formatted_now = now.strftime("%d-%m-%Y %H:%M:%S")
            results = {'result':'error',
                       'bids_basename':bids_path.basename,
                       "date":formatted_now,
                       "details":f"Exception: {str(e)}, {traceback.format_exc()}"
                       }

    else:

        # Not accepted type to process
        now = dt.now(timezone.utc)
        formatted_now = now.strftime("%d-%m-%Y %H:%M:%S")
        results = {'result':'error',
                   'file':bids_path.basename,
                   'details':'Not accepted type of file to process',
                   'date':formatted_now
                   }

    return results



def eeg_artifact_detection(config,bids_path):
    """

    Call the artifact detection proceesses one by one for each type of recording.

    :arg
    config (dict): Configuration parameters (paths, parameters, etc)
    bids_path (BIDSpath): Metadata to process

    :returns
    The annotations.

    """

    # Estimate the SOBI components to detect artifacts
    estimate_artifact_components(config, bids_path, 'sobi_artifacts')

    # Muscle and sensor artifact detection.
    # Muscle
    muscle_annotations = muscle_detection(
        config,
        bids_path)

    # Save the muscle annotations for better jump detection
    _ = bids.write_annot(bids_path,muscle_annotations)

    # Sensor (jumps)
    sensor_annotations = sensor_detection(
        config,
        bids_path)

    # Combine the annotations
    annotations = muscle_annotations.__add__(sensor_annotations)

    # Save the annotations in BIDS format
    _ = bids.write_annot(bids_path, annotations)

    # Estimate the SOBI components to detect artifacts
    estimate_artifact_components(config,bids_path,'sobi')

    # I have to find again all the artifacts
    # For EEG
    # Muscle
    muscle_annotations = muscle_detection(
        config,
        bids_path)

    # Save the muscle annotations for better jump detection
    _ = bids.write_annot(bids_path,muscle_annotations)

    # Sensor (jumps)
    sensor_annotations = sensor_detection(
        config,
        bids_path)

    # EOG
    # Select the frontal channels
    EOG_annotations = EOG_detection(
        config,
        bids_path,
        frontal_channels=config['artifact_detection']["frontal_channels"])

    # Sensor (jumps)
    other_annotations = other_detection(
        config,
        bids_path)

    # Merge all the annotations into a MNE Annotation object
    annotations = other_annotations.__add__(EOG_annotations).__add__(muscle_annotations).__add__(sensor_annotations)

    # Save the annotations in BIDS format
    _ = bids.write_annot(bids_path,annotations)

    return annotations



def estimate_artifact_components(config,bids_path,derivatives_label):
    """

    Estimate ICA using SOBI algorithm after excluding badchannels. Then label each component.
    Save the mixing matrix, the unmixing matrix and the labels in BIDS format.

    :arg
        config (dict): Configuration parameters (paths, parameters, etc)
        bids_path (dict): Path to the recording

    """

    # Parameters for loading EEG recordings
    freq_limits = [config['component_estimation']['low_freq'],
                   config['component_estimation']['high_freq']]
    resample_frequency = config['component_estimation']['resampled_frequency']
    epoch_definition = config['component_estimation']['epoch_definition']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]

    # Include artifacts or not (depending on the current SOBI estimation)
    if derivatives_label == 'sobi_artifacts':
        set_annotations = False
    else:
        set_annotations = True

    # Load raw EEG
    raw = mnetools.prepare_raw(
        config,
        bids_path,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        resample_frequency=resample_frequency,
        freq_limits=freq_limits,
        exclude_badchannels=True,
        set_annotations=set_annotations,
        epoch=epoch_definition,
        rereference=True)

    # Run SOBI
    sobi = mnetools.sobi(raw)

    # Label de ICs
    _ = iclabel.label_components(raw, sobi, method='iclabel')

    # Check the probabilities max probabilities of each category and find those under 0.7
    max_probability = sobi.labels_scores_.max(axis=1)
    index_unclear = [i for i in range(len(max_probability)) if
                     max_probability[i] < config['component_estimation']['unclear_threshold']]

    # Re-arrange the label matrix and assign "other" to the unclears
    labels = ['brain', 'muscle', 'eog', 'ecg', 'line_noise', 'ch_noise']
    for current_label in labels:
        sobi.labels_[current_label] = [i for i in sobi.labels_[current_label] if i not in index_unclear]

    dummy = sobi.labels_['other'].copy()
    dummy.extend(index_unclear)
    dummy = list(set(dummy))
    sobi.labels_['other'] = dummy.copy()

    # Writes the SOBI data into the derivatives folder.
    _ = bids.write_sobi (bids_path, sobi, derivatives_label)



def EOG_detection(config,bids_path, frontal_channels='all'):
    """

    Detect EOG artifacts

    :arg
    config (dict): Configuration parameters (paths, parameters, etc)
    bids_path (dict): Path to the recording
    channels_to_include (str): Channels type to work on.

    :returns
    MNE annotation object with detected EOG artifacts

    """

    # Find EOG aritfacts index
    EOG_index,last_sample,sfreq = find_artifacts.EOG_detection(
        config,
        bids_path,
        frontal_channels=frontal_channels)

    # If any artifact
    if len(EOG_index) > 0:

        # Create the annotations
        EOG_annotations = find_artifacts.create_annotations(
            EOG_index,
            last_sample,
            sfreq,
            'bad_EOG')

    else:

        # Create an empty Annotation
        EOG_annotations = mne.Annotations(onset=[], duration=[], description=[])

    return EOG_annotations



def muscle_detection(config,bids_path):
    """

    Detect muscle artifacts

    :arg
    config (dict): Configuration parameters (paths, parameters, etc)
    bids_path (dict): Path to the recording
    channels_to_include (str): Channels type to work on.

    :returns
    MNE annotation object with detected muscle artifacts

    """

    # Look the position of muscular artifacts
    muscle_index, last_sample, sfreq = find_artifacts.muscle_detection(
        config,
        bids_path)

    # If any index
    if len(muscle_index) > 0:

        # Create the annotations
        muscle_index.sort()
        muscle_annotations = find_artifacts.create_annotations(
            muscle_index,
            last_sample,
            sfreq,
            'bad_muscle',
            fictional_artifact_duration=0.5)

    else:

        # If no artifacts, create empty Annotation
        muscle_annotations = mne.Annotations(onset=[], duration=[], description=[])

    return muscle_annotations



def sensor_detection(config,bids_path):
    """

    Detect sensor artifacts (jumps)

    :arg
    config (dict): Configuration parameters (paths, parameters, etc)
    bids_path (dict): Path to the recording
    channels_to_include (str): Channels type to work on.

    :returns
    MNE annotation object with detected sensor artifacts

    """

    # Look the position of muscular artifacts
    sensor_index,last_sample,sfreq = find_artifacts.sensor_detection(
        config,
        bids_path)

    # Create as Annotations
    if len(sensor_index) > 0:

        # Create the annotations
        sensor_index.sort()  # First sort the list
        sensor_annotations = find_artifacts.create_annotations(
            sensor_index,
            last_sample,
            sfreq,
            'bad_jump',
            fictional_artifact_duration=0.3)

    # If no artifacts, create empty Annotation
    else:
        sensor_annotations = mne.Annotations(onset=[], duration=[], description=[])

    return sensor_annotations



def other_detection(config,bids_path):
    """

    Detect sensor artifacts (jumps)

    :arg
    config (dict): Configuration parameters (paths, parameters, etc)
    bids_path (dict): Path to the recording
    channels_to_include (str): Channels type to work on.

    :returns
    MNE annotation object with detected sensor artifacts

    """

    # Look the position of muscular artifacts
    other_index,last_sample,sfreq = find_artifacts.other_detection(
        config,
        bids_path)

    # Create as Annotations
    if len(other_index) > 0:

        # Create the annotations
        other_index.sort()  # First sort the list
        sensor_annotations = find_artifacts.create_annotations(
            other_index,
            last_sample,
            sfreq,
            'bad_other',
            fictional_artifact_duration=0.3)

    # If no artifacts, create empty Annotation
    else:
        sensor_annotations = mne.Annotations(onset=[], duration=[], description=[])

    return sensor_annotations


