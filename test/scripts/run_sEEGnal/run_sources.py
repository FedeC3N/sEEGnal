#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scripts to call sources_reconstruction

Federico Ramírez-Toraño
11/02/2026

"""

# Imports
from test.init.init import init
from sEEGnal.tools.mne_tools import prepare_eeg
from sEEGnal.tools.bids_tools import create_bids_path
from sEEGnal.sources_reconstruction.forward import make_forward_model
from sEEGnal.sources_reconstruction.covariance import estimate_covariance
from sEEGnal.sources_reconstruction.inverse import estimate_inverse_solution

# Init the database
config, files, sub, ses, task = init()

# List of subjects with errors
errors = []

# Go through each subject
index = [10]
for current_index in index:

    # current info
    current_file = files[current_index]
    current_sub = sub[current_index]
    current_ses = ses[current_index]
    current_task = task[current_index]
    print(current_file)

    # Create the subjects following AI-Mind protocol
    bids_path = create_bids_path(config, current_sub, current_ses, current_task)

    # Get the forward model
    forward_model = make_forward_model(config,bids_path)

    # Compute the covariance matrix
    data_cov = estimate_covariance(config,bids_path)

    # Estimate inverse solution
    filters = estimate_inverse_solution(config,bids_path,
                                        forward_model=forward_model,
                                        data_cov=data_cov)

    # Let's plot sources
    import mne
    # Load the clean EEG
    sobi = {
        'desc': 'sobi',
        'components_to_include': ['brain', 'other'],
        'components_to_exclude': []
    }
    freq_limits = [
        config['component_estimation']['low_freq'],
        config['component_estimation']['high_freq']
    ]
    crop_seconds = config['component_estimation']['crop_seconds']
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]
    epoch_definition = config['source_reconstruction']['epoch_definition']

    # Load the clean data
    raw = prepare_eeg(
        config,
        bids_path,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        freq_limits=freq_limits,
        notch_filter=True,
        resample_frequency=resample_frequency,
        set_annotations=True,
        crop_seconds=crop_seconds,
        rereference='average'
    )

    raw = prepare_eeg(
        config,
        bids_path,
        raw=raw,
        apply_sobi=sobi,
        freq_limits=[2, 45],
        metadata_badchannels=True,
        epoch_definition=epoch_definition
    )

    # Get sources time series
    sources_time_series = mne.beamformer.apply_lcmv_epochs(raw, filters)

    # Plot
    import matplotlib.pyplot as plt
    import numpy as np
    data = np.zeros((sources_time_series[0].shape[0],sources_time_series[
        0].shape[1],len(sources_time_series)))
    for isrc in sources_time_series:
        vsrc = sources_time_series[5]
        data = np.mean(vsrc.data,axis=0)  # shape: n_times
        sfreq = raw.info['sfreq']  # sampling frequency used for beamformer

        psd, freqs = mne.time_frequency.psd_array_welch(
            data,
            sfreq=sfreq,
            fmin=2, fmax=45,
            n_fft=256)

    plt.figure()
    plt.semilogy(freqs, psd)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.show()