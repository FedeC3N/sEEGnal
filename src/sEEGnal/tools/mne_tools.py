#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 12:40:25 2023

@author: bru
"""

import re
import datetime

import mne
import numpy
import scipy.signal

import sEEGnal.tools.bss as bss
import sEEGnal.tools.tools as tools
import sEEGnal.tools.signal as signal
import sEEGnal.tools.spheres as spheres
import sEEGnal.tools.bids_tools as bids
from sEEGnal.io.read_source_files import read_source_files

# Lists the valid MNE objects.
mnevalid = (mne.io.BaseRaw, mne.BaseEpochs)

# Sets the verbosity level for MNE.
mne.set_log_level(verbose='ERROR')


# Function for two-pass filtering on MNE objects.
def filtfilt(mnedata, num=1, den=1, hilbert=False):
    """ Wrapper to apply two-pass filtering to MNE objects."""

    # Checks if the data is a valid MNE object.
    if not isinstance(mnedata, mnevalid):

        print('Unsupported data type.')
        return None

    # Creates a copy of the input data.
    mnedata = mnedata.copy()

    # Gets the raw data matrix.
    rawdata = mnedata.get_data()

    # For IIR filters uses SciPy (faster and more accurate).
    if numpy.array(den).size != 1:

        # Gets the data metadata.
        dshape = rawdata.shape
        nsample = dshape[-1]

        # Reshapes the data into a 2D array.
        rawdata = rawdata.reshape((-1, nsample))

        # Filters the data.
        rawdata = scipy.signal.filtfilt(num, den, rawdata)

        # Restores the original data shape.
        rawdata = rawdata.reshape(dshape)

    # For FIR filters use FFT (much faster, same accuracy).
    else:

        # Filters the data.
        rawdata = signal.filtfilt(rawdata, num=num, den=den, hilbert=hilbert)

    # Replaces the data and marks it as loaded.
    mnedata._data = rawdata
    mnedata.preload = True

    ## Creates a new MNE object with the filtered data.
    #mnedata   = mne.EpochsArray ( rawdata, data.info, events = data.events, verbose = False )

    ## Creates a new MNE object with the filtered data.
    #mnedata    = mne.io.RawArray ( rawdata, data.info, verbose = False )

    # Returns the MNE object.
    return mnedata


def decimate(mnedata, ratio=1):
    """Decimates an MNE object with no filtering."""
    """
    Based on MNE 1.7 functions:
    mne.BaseRaw.resample
    https://github.com/mne-tools/mne-python/blob/maint/1.7/mne/io/base.py
    mne.Epochs.decimate
    https://github.com/mne-tools/mne-python/blob/maint/1.7/mne/utils/mixin.py
    """

    # Checks if the data is a valid MNE object.
    if not isinstance(mnedata, mnevalid):

        print('Unsupported data type.')
        return None

    # Ratio is 1 does nothing.
    if ratio == 1:
        return mnedata

    # Creates a copy of the input data.
    mnedata = mnedata.copy()

    # Gets the raw data matrix.
    rawdata = mnedata.get_data()

    # Decimates the raw data matrix in the last dimension.
    decdata = rawdata[..., ::ratio]

    # Replaces the data and marks it as loaded.
    mnedata._data = decdata
    mnedata.preload = True

    # Updates the sampling rate.
    with mnedata.info._unlock():
        mnedata.info['sfreq'] = mnedata.info['sfreq'] / ratio

    # Updates the mne.Raw information.
    if isinstance(mnedata, mne.io.BaseRaw):
        n_news = numpy.array(decdata.shape[1:])
        mnedata._cropped_samp = int(numpy.round(mnedata._cropped_samp * ratio))
        mnedata._first_samps = numpy.round(mnedata._first_samps * ratio).astype(int)
        mnedata._last_samps = numpy.array(mnedata._first_samps) + n_news - 1
        mnedata._raw_lengths[:1] = list(n_news)

    # Updates the mne.Epochs information.
    if isinstance(mnedata, mne.BaseEpochs):
        mnedata._decim = 1
        mnedata._set_times(mnedata._raw_times[::ratio])
        mnedata._update_first_last()

    # Returns the MNE object.
    return mnedata


def fixchan(mnedata, elec=None):
    """Wrapper to apply spherical splines channel reconstruction to MNE objects."""

    # Checks if the data is a valid MNE object.
    if not isinstance(mnedata, mnevalid):

        print('Unsupported data type.')
        return None

    # Creates a copy of the input data.
    mnedata = mnedata.copy()

    # If no bad channels does nothing.
    if len(mnedata.info['bads']) == 0:
        return mnedata

    # If no electrode definition provided, loads the 10-05 default montage.
    if elec is None:
        elec = mne.channels.make_standard_montage('standard_1005')
        elec = elec.get_positions()['ch_pos']

    # Generates the reduced montage for the data.
    elec = {ch: elec[ch] for ch in elec.keys() if ch in mnedata.ch_names}

    # Generates the reduced montage for the good channels.
    elec1 = {ch: elec[ch] for ch in elec.keys() if ch not in mnedata.info['bads']}

    # Generates the reduced montage for the bad channels.
    elec2 = {ch: elec[ch] for ch in elec.keys() if ch in mnedata.info['bads']}

    # If no bad channels returns the untouched data.
    if len(elec2) == 0:
        return mnedata

    # Gets the reconstruction matrix.
    wPot = spheres.spline_int(elec1, elec2)

    # Gets the data-to-data transformation matrix.
    d2d = numpy.eye(len(mnedata.ch_names))

    # Gets the indexes of the good and bad channels.
    hits1 = tools.find_matches(list(elec1.keys()), mnedata.ch_names)
    hits2 = tools.find_matches(list(elec2.keys()), mnedata.ch_names)

    # Zeroes the bad channels.
    d2d[hits2, hits2] = 0

    # Adds the reconstruction mapping for the bad channels.
    d2d[numpy.ix_(hits2, hits1)] = wPot

    # Gets the raw data matrix.
    #rawdata = mnedata.get_data ( copy = True )
    rawdata = mnedata.get_data()
    """
    # Rewrites the raw data as epochs * samples * channels.
    shape   = rawdata.shape
    tmpdata = rawdata.reshape ( ( -1, ) + shape [ -2: ] )
    tmpdata = rawdata.swapaxes ( -1, -2 )

    # Fixes all the epochs at once.
    fixdata = numpy.dot ( tmpdata, d2d.T )

    # Restores the original data shape.
    fixdata = fixdata.swapaxes ( -1, -2 )
    fixdata = fixdata.reshape ( shape )
    """

    # Rewrites the raw data as epochs * channels * samples.
    shape = rawdata.shape
    tmpdata = rawdata.reshape((-1, ) + shape[-2:])

    # Fixes the bad channels epoch-wise.
    fixdata = numpy.zeros(tmpdata.shape)
    for i in range(tmpdata.shape[0]):
        fixdata[i] = numpy.dot(d2d, tmpdata[i])

    # Restores the original data shape.
    fixdata = fixdata.reshape(shape)

    # Updates the MNE object.
    mnedata._data = fixdata

    # Returns the fixed MNE object.
    return mnedata


# Perform SOBI on MNE object
def sobi(mnedata, nlag=None, nsource=None):
    ''' Wrapper to estimating SOBI componentes from MNE objects.'''

    # Checks if the data is a valid MNE object.
    if not isinstance(mnedata, mnevalid):

        print('Unsupported data type.')
        return None

    # Creates a copy of the input data.
    mnedata = mnedata.copy()

    # Gets the channel labels.
    chname = mnedata.ch_names

    # Gets the raw data matrix.
    rawdata = mnedata.get_data()

    # Estimates the SOBI mixing matrix.
    mixing, unmixing = bss.sobi(rawdata, nlag=nlag, nsource=nsource)

    # Builds the MNE ICA object.
    mnesobi = build_bss(mixing, unmixing, chname, method='sobi')

    # Returns the MNE ICA object.
    return mnesobi


def build_raw(info, data, montage=None):

    # Lists the channels in the data.
    ch_label = info['channels']['label']

    # If no montage assumes the standard 10-05.
    if montage is None:
        montage = mne.channels.make_standard_montage('standard_1005')

    # Identifies the EEG, EOG, ECG, and EMG channels.
    ind_eeg = numpy.where(numpy.in1d(ch_label, montage.ch_names))
    ind_eog = numpy.where([re.search('EOG', label) != None for label in ch_label])
    ind_ecg = numpy.where([re.search('CLAV', label) != None for label in ch_label])
    ind_emg = numpy.where([re.search('EMG', label) != None for label in ch_label])

    # Marks all the channels as EEG.
    ch_types = numpy.array(['eeg'] * len(ch_label))

    # Sets the channel types.
    ch_types[ind_eeg] = 'eeg'
    ch_types[ind_eog] = 'eog'
    ch_types[ind_ecg] = 'ecg'
    ch_types[ind_emg] = 'emg'

    # Creates the MNE-Python information object.
    mneinfo = mne.create_info(
        ch_names=list(info['channels']['label']), sfreq=info['sample_rate'], ch_types=list(ch_types)
    )

    # Adds the montage, if provided.
    if montage is not None:
        mneinfo.set_montage(montage)

    # Creates the MNE-Python raw data object.
    mneraw = mne.io.RawArray(data.T, mneinfo, verbose=False)

    # Overwrites the default parameters.
    mneraw.set_meas_date(info['acquisition_time'])

    # Adds the calibration factor.
    mneraw._cals = numpy.ones(len(ch_label))

    # Marks the 'active' channels.
    mneraw._read_picks = [numpy.arange(len(ch_label))]

    # Gets the information about the impedances, if any.
    if 'impedances' in info:

        # Takes only the first measurement.
        if len(info['impedances']) > 0:
            impmeta = info['impedances'][0]
            impedances = impmeta['measurement']

            # Fills the extra information for MNE.
            for channel, value in impedances.items():

                impedances[channel] = {
                    'imp': value,
                    'imp_unit': impmeta['unit'],
                    'imp_meas_time': datetime.datetime.fromtimestamp(impmeta['time'])
                }

            # Adds the impedances to the MNE object.
            mneraw.impedances = impedances

    # Gets the annotations, if any.
    annotations = mne.Annotations(
        [annot['onset'] for annot in info['events']], [annot['duration'] for annot in info['events']],
        [annot['description'] for annot in info['events']]
    )

    # Adds the annotations to the MNE object.
    mneraw.set_annotations(annotations)

    # Returns the MNE Raw object.
    return mneraw


def build_bss(mixing, unmixing, chname, icname=None, method='bss'):

    # Gets the number of channels and components.
    nchannel = mixing.shape[0]
    nsource = mixing.shape[1]

    # Generates the labels for the components, if no provided.
    if icname is None:

        # Creates the labels.
        icname = ['IC%03d' % (index + 1) for index in range(nsource)]

    # If the matrices are not square adds some dummy components.
    if nchannel != nsource:

        # Creates the dummy components.
        ndummy = nchannel - nsource
        mdummy = numpy.zeros([nchannel, ndummy])

        # Creates the dummy labels.
        ldummy = ['DUM%03d' % (index + 1) for index in range(ndummy)]

        # Concatenates the matrix and the labels.
        mixing = numpy.append(mixing, mdummy, 1)
        unmixing = numpy.append(unmixing, mdummy.T, 0)
        icname = icname + ldummy

    # Creates a dummy MNE ICA object.
    mnebss = mne.preprocessing.ICA()

    # Fills the object with the SOBI data.
    mnebss.ch_names = chname
    mnebss._ica_names = icname
    mnebss.mixing_matrix_ = mixing
    mnebss.unmixing_matrix_ = unmixing

    # Fills some dummy metadata.
    mnebss.current_fit = 'raw'
    mnebss.method = method
    mnebss.n_components_ = mixing.shape[0]
    mnebss.n_iter_ = 0
    mnebss.n_samples_ = 0
    mnebss.pca_components_ = numpy.eye(nchannel)
    mnebss.pca_explained_variance_ = numpy.ones(nchannel)
    mnebss.pca_mean_ = numpy.zeros(nchannel)
    mnebss.pre_whitener_ = 0.01 * numpy.ones([nchannel, 1])

    # Returns the MNE ICA object.
    return mnebss


# Function to segment and MNE raw data object avoiding the artifacts.
def get_epochs(raw, annot=None, length=4, overlap=None, padding=None, preload=False):
    # Gets the annotations from the raw data, if required.
    if annot is None:
        annot = raw.annotations

    # Removes the annotations extending beyond the data.
    annot.crop(tmin=raw.times[0], tmax=raw.times[-1])

    # Sanitizes the input.
    overlap = overlap or 0
    padding = padding or 0

    # Gets the first and last valid times.
    ftime = raw.times[0] + padding
    ltime = raw.times[-1] - padding - overlap

    # Gets the separation between epochs.
    step = length - overlap

    # Gets the beginning and ending indexes for each artifact.
    artbeg = annot.onset
    artend = annot.onset + annot.duration

    # Merges overlapping artifacts.
    for index in range(len(artbeg) - 1, 0, -1):

        # Checks if the artifacts overlap.
        if artend[index - 1] > artbeg[index]:
            # Extendes the previous artifact, if required.
            artend[index - 1] = numpy.max(artend[index - 1:index + 1])

            # Removes the current artifact.
            artbeg = numpy.delete(artbeg, index)
            artend = numpy.delete(artend, index)

    # Adds virtual artifacts at the beginning and end of the data.
    artbeg = numpy.append(artbeg, ltime)
    artend = numpy.insert(artend, 0, ftime)

    # Lists the clean segments of data.
    segbegs = list(artend)
    seglens = list(artbeg - artend)

    # Initializes the list of epoch onsets.
    onsets = []

    # Iterates through each clean segment.
    for index in range(len(segbegs)):
        # Splits the segment in as many epochs as possible.
        nepoch = numpy.floor((seglens[index] - length) / step).astype(int) + 1
        sides = ((seglens[index] - length) - step * (nepoch - 1)) / 2

        # Creates the markers for the epochs.
        onset = segbegs[index] + sides + step * numpy.array(range(nepoch))

        # Adds the epochs to the list.
        onsets = onsets + list(onset)

    # Creates a series of events from the defined epochs.
    events = numpy.zeros([len(onsets), 3], 'int')
    events[:, 0] = raw.time_as_index(onsets)
    events[:, 2] = 1

    # Generates a MNE epoch structure from the data and the events.
    epochs = mne.Epochs(
        raw, events, tmin=-padding, tmax=length + padding, baseline=None, verbose=False, preload=preload
    )

    # Returns the epochs object.
    return epochs


# Function to prepare MNE raw data
def prepare_eeg(
    config,
    bids_path,
    preload=False,
    channels_to_include=['all'],
    channels_to_exclude=[],
    apply_sobi = False,
    resample_frequency=False,
    notch_filter=False,
    freq_limits=None,
    crop_seconds=None,
    exclude_badchannels=False,
    interpolate_bads=False,
    set_annotations=False,
    rereference=False,
    epoch=None
):

    ##################################################################
    # Load EEG and montage
    ##################################################################

    # Read the file
    raw = read_source_files(str(bids_path.fpath),preload=preload)

    # Set montage
    raw.set_montage('standard_1005', on_missing='ignore')

    # Include and exclude channels explicitly
    raw.pick(channels_to_include)
    raw.drop_channels(channels_to_exclude, on_missing='ignore')

    ##################################################################
    # Include / exclude components
    ##################################################################

    if apply_sobi:

        # Load the SOBI
        # Read the ICA information
        sobi = bids.read_sobi(bids_path, apply_sobi['desc'])

        # IClabel recommend to filter between 1 and 100
        raw.filter(1, 100)

        # Select the components and remove
        IClabel_componets = ['brain', 'muscle', 'eog', 'ecg', 'line_noise',
                             'ch_noise', 'other']
        include = set(apply_sobi['components_to_include'] or IClabel_componets)
        exclude = set(apply_sobi['components_to_exclude'] or [])
        final_inclusion_list = list(include - exclude)

        # If any to apply
        if len(IClabel_componets) != len(final_inclusion_list):

            final_inclusion_index = [sobi.labels_[current_label] for
                                     current_label in final_inclusion_list]
            final_inclusion_index = sum(final_inclusion_index, [])

            # Apply
            sobi.apply(raw, include=final_inclusion_index)

    ##################################################################
    # Downsample
    ##################################################################

    if resample_frequency:
        raw.resample(resample_frequency)

    ##################################################################
    # Filters
    ##################################################################

    # Remove 50 Hz noise (and harmonics)
    if notch_filter:

        # Check the maximum frequency I can use
        notch_freqs = [current_freq for current_freq in config['component_estimation']['notch_frequencies']
                       if current_freq < resample_frequency/2 ]

        # Apply the filter if any
        if len(notch_freqs) > 0:
            raw.notch_filter(notch_freqs)

    # Filter the data
    if freq_limits:
        if len(freq_limits) != 2:
            raise ValueError(
                "You need to define two frequency limits to filter")
            # print('You need to define two frequency limits to filter')
        else:

            # Check the maximum frequency I can use
            if freq_limits[1] > raw.info['sfreq']/2:
                freq_limits[1] = raw.info['sfreq']/2 - 0.01

            raw.filter(freq_limits[0], freq_limits[1])

    ##################################################################
    # Remove bad channels
    ##################################################################

    if exclude_badchannels:

        chan = bids.read_chan(bids_path)
        badchannels = list(chan.loc[chan['status'] == 'bad']['name'])

        # To avoid errors, the badchannels has to be among the channels included in the recording
        badchannels = list(set(badchannels) & set(raw.info['ch_names']))
        raw.info['bads'] = badchannels

        # If all channels are badchannels, raise an Exception
        if (len(badchannels) == len(raw.info['ch_names'])):
            raise Exception("All channels are marked as bad")

        # Drop or interpolate the badchannels
        if interpolate_bads == True:
            raw.interpolate_bads()
        else:
            raw = raw.pick(None, exclude='bads')

    ##################################################################
    # Add the annotations if requested
    ##################################################################

    if set_annotations:

        # Reads the annotations.
        annotations = bids.read_annot(bids_path)

        # Adds the annotations to the MNE object.
        if len(annotations):
            raw.set_annotations(annotations)

    ##################################################################
    # Crop (in seconds)
    ##################################################################

    if crop_seconds:
        raw.crop(tmin=crop_seconds, tmax=raw.times[-1] - crop_seconds)

    ##################################################################
    # Set reference
    ##################################################################

    # Average
    if rereference == 'average':
        raw.set_eeg_reference(ref_channels='average')

    # Median
    if rereference == 'median':
        # Get the data (shape: n_epochs × n_channels × n_times)
        data = raw.get_data()

        # Compute the median across channels
        median_ref = numpy.median(data, axis=1, keepdims=True)

        # Subtract the median reference from all channels
        data -= median_ref

        # Put the rereferenced data back into the epochs object
        raw._data = data

    ##################################################################
    # Epoch the data
    ##################################################################

    if epoch:
        if ('length' in epoch.keys()) and ('overlap' in epoch.keys()) and ('padding' in epoch.keys()):
            raw = get_epochs(
                raw,
                annot=raw.annotations,
                length=epoch['length'],
                overlap=epoch['overlap'],
                padding=epoch['padding'],
                preload=preload
            )
        else:
            raise ValueError("Dictionary definition must be {'length':[],'overlap':[],'padding':[]}")

    return raw
