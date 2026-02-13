# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 14:15:56 2022

@author: Ricardo
"""

from functools import wraps
import shutil
import inspect
import pandas
import json
import os

import mne
import mne_bids

import sEEGnal.tools.mne_tools as mnetools
import sEEGnal.tools.tools as tools
import sEEGnal.tools.tsv_tools as tsv



def init_derivatives(func):
    """

    Check if the BIDS structure exists and if not, create it.

    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        # Get the parameteres
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        config = bound.arguments.get('config')
        BIDS = bound.arguments.get('BIDS')

        # Check if the derivatives exists
        # Creates the paths to the raw and derivative datas.
        derivatives_file_path = build_derivative(BIDS, config['subsystem'], 'channels.tsv')
        derivatives_folder_path = os.path.dirname(derivatives_file_path)

        # Creates the derivatives folder, if required.
        if not os.path.isdir(derivatives_folder_path):

            os.makedirs(derivatives_folder_path)

            # Add a dummy file
            bids_file_path = build_bids(BIDS, 'channels.tsv')
            shutil.copy(bids_file_path, derivatives_file_path)

        # Continue with the call
        created_files = func(*args, **kwargs)

        return created_files

    return wrapper


@init_derivatives
def write_annotations(config,BIDS, annotations=None):

    # Initializes the list of created files.
    created_files = []

    # If no annotation, creates an empty one.
    if annotations is None:
        annotations = mne.Annotations([], [], [])

    # Initializes the annotations.
    tsv_data = pandas.DataFrame(columns=['onset', 'duration', 'label'])

    # Adds the annotations to the data frame.
    tsv_data['onset'] = annotations.onset
    tsv_data['duration'] = annotations.duration
    tsv_data['label'] = annotations.description

    # Creates the JSON dictionary.
    json_data = {
        'label': {
            'LongName': 'Artifacts information.',
            'Description': 'Annotations defining the artifacts present in the data.',
            'Levels': {
                'onset': 'Second of onset',
                'duration': 'Duration of the artifact.',
                'label': 'Kind of artifact'
            }
        }
    }

    # Builds the path to the files.
    tsv_file = build_derivative(BIDS, 'preprocess','desc-artifacts_annotations.tsv')
    json_file = build_derivative(BIDS, 'preprocess','desc-artifacts_annotations.json')

    # Writes the data.
    tsv.write_tsv(tsv_data, tsv_file)

    # Writes the JSON dictionary.
    with open(json_file, 'w') as fp:
        json.dump(json_data, fp, indent=4)

    # Appends the created file to the list.
    created_files.append(tsv_file)
    created_files.append(json_file)

    return created_files


def read_annotations(config,BIDS):

    # Builds the path to the file.
    tsv_file = build_derivative(BIDS, 'preprocess', 'desc-artifacts_annotations.tsv')

    # Loads the derivative data pieces.
    if tsv_file.exists():
        tsv_data = tsv.read_tsv(tsv_file, ismatrix=False)

        # Builds the MNE annotation object.
        annotations = mne.Annotations(tsv_data['onset'], tsv_data['duration'],
                                tsv_data['label'])

    else:
        annotations = mne.Annotations([], [], [])

    # Returns the annotations object.
    return annotations


def read_badchannels(config,BIDS):

    # Builds the path to the file.
    tsv_file = build_derivative(BIDS, 'preprocess','channels.tsv')

    # Loads the derivative data pieces.
    channels = tsv.read_tsv(tsv_file, ismatrix=False)

    # Returns the channels table object.
    return channels


@init_derivatives
def update_badchans(config,BIDS, badchannels=None, badchannels_description=None):

    # Builds the path to the file.
    tsv_file = build_derivative(BIDS, 'preprocess','channels.tsv')

    # Reads the contents of the raw file.
    tsv_data = mne_bids.tsv_handler._from_tsv(tsv_file)

    # Identifies the bad channels.
    badindex = tools.find_matches(badchannels, tsv_data['name'])

    # Update the values
    for ibad in range(len(badindex)):

        # Update the status
        tsv_data['status'][badindex[ibad]] = 'bad'

        # If it is the first change in description, change it. If not, append it.
        current_status = tsv_data['status_description'][badindex[ibad]]
        if current_status == 'n/a':
            tsv_data['status_description'][badindex[ibad]] = badchannels_description[ibad]
        else:
            current_status = current_status + ',' + badchannels_description[ibad]
            tsv_data['status_description'][badindex[ibad]] = current_status

    # Saves the TSV file.
    mne_bids.tsv_handler._to_tsv(tsv_data, tsv_file)


@init_derivatives
def write_sobi(config,BIDS, sobi, desc='sobi'):

    # Initializes the list of created files.
    created_files = []

    # Formats the mixing matrix.
    tsv_data = {'matrix': sobi.mixing_matrix_, 'rows': sobi.ch_names, 'columns': sobi._ica_names}

    # Creates the JSON dictionary.
    json_data = {
        "Description":
        "Mixing matrix for Independent Component analysis. Rows represent channels. Columns represent independent components.",
        "IntendedFor": "",
        "Algorithm": "Second Order Blind Identification (SOBI)",
        "Software":
        "See: Belouchrani, A., et al. \"Proceedings of the International Conference on Digital Signal Processing.\" (1993): 346-351.",
        "PCAComponents": "n/a",
        "Seed": "n/a"
    }

    # Builds the path to the files.
    tsv_file = build_derivative(BIDS,'preprocess', 'desc-' + desc + '_mixing.tsv')
    json_file = build_derivative(BIDS,'preprocess', 'desc-' + desc + '_mixing.json')

    # Writes the data.
    tsv.write_tsv(tsv_data, tsv_file)

    # Writes the JSON dictionary.
    with open(json_file, 'w') as fp:
        json.dump(json_data, fp, indent=4)

    # Appends the created file to the list.
    created_files.append(tsv_file)
    created_files.append(json_file)

    # Formats the unmixing matrix.
    tsv_data = {'matrix': sobi.unmixing_matrix_, 'rows': sobi._ica_names, 'columns': sobi.ch_names}

    # Creates the JSON dictionary.
    json_data = {
        "Description":
        "Mixing matrix for Independent Component analysis. Rows represent channels. Columns represent independent components.",
        "IntendedFor": "",
        "Algorithm": "Second Order Blind Identification (SOBI)",
        "Software":
        "See: Belouchrani, A., et al. \"Proceedings of the International Conference on Digital Signal Processing.\" (1993): 346-351.",
        "PCAComponents": "n/a",
        "Seed": "n/a"
    }

    # Builds the path to the files.
    tsv_file = build_derivative(BIDS, 'preprocess','desc-' + desc + '_unmixing.tsv')
    json_file = build_derivative(BIDS, 'preprocess','desc-' + desc + '_unmixing.json')

    # Writes the data.
    tsv.write_tsv(tsv_data, tsv_file)

    # Writes the JSON dictionary.
    with open(json_file, 'w') as fp:
        json.dump(json_data, fp, indent=4)

    # Appends the created file to the list.
    created_files.append(tsv_file)
    created_files.append(json_file)

    # Writes the labels, if any.
    if hasattr(sobi, 'labels_'):

        # Initializes the annotation table.
        tsv_data = pandas.DataFrame(columns=['onset', 'duration', 'label', 'channel'])

        # Adds the annotations.
        for key in sobi.labels_.keys():

            # List the components with the current label.
            hits = [sobi._ica_names[ind] for ind in sobi.labels_[key]]

            # Joins the components using the separator.
            hits = ','.join(hits)

            if not hits:
                hits = None

            # Generates a new table for the current label.
            new_data = pandas.DataFrame.from_dict({'label': [key], 'channel': [hits]})

            # Adds the new entry to the annotations.
            tsv_data = pandas.concat([tsv_data, new_data])

        # Creates the JSON dictionary.
        json_data = {
            'label': {
                'LongName': 'Label of the artifact.',
                'Description': 'Annotations defining the type of artifact present in the data.',
                'Levels': {
                    'brain': 'True brain signal',
                    'muscle': 'Component containing muscular contractions.',
                    'eog': 'Component containing blink-related or eye-movement-related artifacts.',
                    'ecg': 'Component containing heart-related artifacts.',
                    'line_noise': 'Component containing line noise.',
                    'ch_noise': 'Component containing jump artefacts.',
                    'other': 'Component containing other types of noise.'
                }
            },
            'channel': {
                'LongName': 'Components with this label.',
                'Description': 'List of components with this label, if any.',
                'Delimiter': ','
            }
        }

        # Builds the path to the files.
        tsv_file = build_derivative(BIDS, 'preprocess','desc-' + desc + '_annotations.tsv')
        json_file = build_derivative(BIDS, 'preprocess','desc-' + desc + '_annotations.json')

        # Writes the data.
        tsv.write_tsv(tsv_data, tsv_file)

        # Writes the JSON dictionary.
        with open(json_file, 'w') as fp:
            json.dump(json_data, fp, indent=4)

        # Appends the created file to the list.
        created_files.append(tsv_file)
        created_files.append(json_file)

        # Writes the label scores, if any.
        if hasattr(sobi, 'labels_scores_'):

            # Formats the scores matrix.
            tsv_data = {'matrix': sobi.labels_scores_, 'rows': sobi._ica_names, 'columns': sobi.labels_.keys()}

            # Creates the JSON dictionary.
            json_data = {
                "Description": "Prediction scores for each component obtained through IClabel. "
                "Rows represent components. Columns represent labels (Brain, Muscle,,Eye,,Heart, Line Noise, Channel Noise, Other ",
                "Software": "IClabel",
            }

            # Builds the path to the files.
            tsv_file = build_derivative(BIDS, 'preprocess','desc-' + desc + '_prediction_scores.tsv')
            json_file = build_derivative(BIDS, 'preprocess', 'desc-' + desc + '_prediction_scores.json')

            # Writes the data.
            tsv.write_tsv(tsv_data, tsv_file)

            # Writes the JSON dictionary.
            with open(json_file, 'w') as fp:
                json.dump(json_data, fp, indent=4)

            # Appends the created file to the list.
            created_files.append(tsv_file)
            created_files.append(json_file)

    return created_files


@init_derivatives
def read_sobi(config,BIDS, desc='sobi'):

    # Loads the derivative data pieces.
    mix_path = build_derivative(BIDS,'preprocess','desc-' + desc + '_mixing.tsv')
    mix = tsv.read_tsv(mix_path, ismatrix=True)
    unmix_path = build_derivative(BIDS,'preprocess','desc-' + desc + '_unmixing.tsv')
    unmix = tsv.read_tsv(unmix_path, ismatrix=True)

    # Extracts the mixing and unmixing matrices and metadata.
    chname_mix = list(mix['rows'])
    icname_mix = list(mix['columns'])
    matrix_mix = mix['matrix'].astype(float)
    icname_unmix = list(unmix['rows'])
    chname_unmix = list(unmix['columns'])
    matrix_unmix = unmix['matrix'].astype(float)

    # Checks the completitude and integrity of the data.
    assert len(chname_mix) == len(set(chname_mix)), 'Duplicated channels in the mixing matrix.'
    assert len(chname_unmix) == len(set(chname_unmix)), 'Duplicated channels in the unmixing matrix.'
    assert set(chname_mix) == set(chname_unmix), 'The channels in the mixing and unmixing matrices are different.'
    assert len(icname_mix) == len(set(icname_mix)), 'Duplicated components in the mixing matrix.'
    assert len(icname_unmix) == len(set(icname_unmix)), 'Duplicated components in the unmixing matrix.'
    assert set(icname_mix) == set(icname_unmix), 'The components in the mixing and unmixing matrices are different.'

    # Finds the matches between the mixing and unmixing matrices.
    chindex = tools.find_matches(chname_unmix, chname_mix)
    icindex = tools.find_matches(icname_unmix, icname_mix)

    # Sorts the unmixing matrix according to the mixing matrix.
    matrix_unmix = matrix_unmix[:, chindex]
    matrix_unmix = matrix_unmix[icindex, :]

    # Builds the MNE ICA object.
    mneica = mnetools.build_bss(matrix_mix, matrix_unmix, chname_mix, icname=icname_mix)

    # Loads the annotations, if available.
    tsv_file = build_derivative(BIDS, 'preprocess','desc-' + desc + '_annotations.tsv')
    json_file = build_derivative(BIDS,'preprocess','desc-' + desc + '_annotations.json')
    if os.path.isfile(tsv_file):

        # Loads the annotations.
        tsv_data = tsv.read_tsv(tsv_file)
        annot = tsv_data

        # Loads the JSON information.
        with open(json_file, 'r') as fp:
            json_data = json.load(fp)

        # Explodes the channels.
        if 'channel' in json_data.keys() and 'Delimiter' in json_data['channel'].keys():

            # Gets the separator.
            sep = json_data['channel']['Delimiter']

            # Splits the list of channels/components.
            annot.channel = [ch.split(sep) if isinstance(ch, str) else [] for ch in annot.channel]

        # Explodes the annotations to get an entry per channel.
        dummy = annot.explode('channel')
        dummy = dummy.query('channel.notna()').reset_index()

        # Checks the completeness and integrity of the data.
        assert set(dummy.channel
                   ).issubset(set(icname_mix)), 'Some bad components are not defined in the mixing/unmixing matrices.'

        # Adds the annotations as component labels to the ICA object.
        for label in annot['label']:
            hits = annot['label'] == label
            vals = annot.loc[hits, 'channel'].to_list()[0]
            hits = tools.find_matches(vals, mneica._ica_names)
            mneica.labels_[label] = hits

        # Loads the label scores, if available.
        tsv_file = build_derivative(BIDS,'preprocess',  'desc-' + desc + '_prediction_scores.tsv')
        if os.path.isfile(tsv_file):

            score = tsv.read_tsv(tsv_file, ismatrix=True)

            # Extracts the score matrix and metadata.
            icname_score = list(score['rows'])
            lname_score = list(score['columns'])
            matrix_score = score['matrix'].astype(float)

            # Finds the matches between the mixing and score matrices.
            icindex = tools.find_matches(icname_score, icname_mix)
            lindex = tools.find_matches(lname_score, annot['label'])

            # Sorts the scores matrix according to the mixing matrix and the annotations.
            matrix_score = matrix_score[:, lindex]
            matrix_score = matrix_score[icindex, :]

            # Adds the prediction scores to the ICA object.
            mneica.labels_scores_ = matrix_score

    # Returns the ICA object.
    return mneica


def build_bids(BIDS, fname_tail):

    from mne_bids.config import ALLOWED_PATH_ENTITIES_SHORT

    # Gets the base filename structure.
    basename = []

    for key, val in BIDS.entities.items():
        if val is not None and key != 'datatype':
            long_to_short_entity = {val: key for key, val in ALLOWED_PATH_ENTITIES_SHORT.items()}
            key = long_to_short_entity[key]
            basename.append(f'{key}-{val}')

    # Builds the file name for the piece of data.
    file_bids = '_'.join(basename + [fname_tail])

    # Builds the complete path to the piece of data.
    path_rel = BIDS.directory.relative_to(BIDS.root)
    path_bids = os.path.join(BIDS.root,path_rel,file_bids)

    return path_bids


def build_derivative(BIDS, process, fname_tail):

    from mne_bids.config import ALLOWED_PATH_ENTITIES_SHORT

    # Gets the base filename structure.
    basename = []

    for key, val in BIDS.entities.items():
        if val is not None and key != 'datatype':
            long_to_short_entity = {val: key for key, val in ALLOWED_PATH_ENTITIES_SHORT.items()}
            key = long_to_short_entity[key]
            basename.append(f'{key}-{val}')

    # Builds the file name for the piece of data.
    file_der = '_'.join(basename + [fname_tail])

    # Builds the complete path to the piece of data.
    path_rel = BIDS.directory.relative_to(BIDS.root)
    path_etl = BIDS.root.joinpath(os.path.join('derivatives', 'sEEGnal',process))
    path_der = path_etl.joinpath(path_rel).joinpath(file_der)

    # Returns the path to the piece of data.
    return path_der


def build_BIDS(config, current_sub, current_ses, current_task):

    # Remove the unallowed characters
    current_sub = current_sub.replace('-', '')
    current_task = current_task.replace('-', '')

    # Builds the BIDS path from the metadata.
    BIDS = mne_bids.BIDSPath(
        subject=current_sub,
        session=current_ses,
        task=current_task,
        datatype='eeg',
        root=config['path']['data_root'],
        suffix='eeg',
        extension='.vhdr'
    )

    return BIDS
