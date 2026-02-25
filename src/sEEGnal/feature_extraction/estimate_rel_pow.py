# -*- coding: utf-8 -*-
"""

Estimate relative power with Welch's method for sensors or sources

Federico Ramírez-Toraño
24/02/2026

"""

# Imports
import numpy

from sEEGnal.tools.mne_tools import prepare_eeg
from sEEGnal.tools.bids_tools import write_relative_psd

def estimate_rel_pow(config,BIDS):

    # Load the clean EEG
    sobi = {
        'desc': 'sobi',
        'components_to_include': ['brain', 'other'],
        'components_to_exclude': []
    }
    freq_limits_components = [
        config['component_estimation']['low_freq'],
        config['component_estimation']['high_freq']
    ]
    freq_limits_signal = [
        config['feature_extraction']['rel_pow']['sensor']['freq_bands_limits'][0][0],
        config['feature_extraction']['rel_pow']['sensor']['freq_bands_limits'][-1][-1]
    ]
    crop_seconds = config['component_estimation']['crop_seconds']
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]
    epoch_definition = config['source_reconstruction']['epoch_definition']

    # Load the clean data
    raw = prepare_eeg(
        config,
        BIDS,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        freq_limits=freq_limits_components,
        notch_filter=True,
        resample_frequency=resample_frequency,
        set_annotations=True,
        crop_seconds=crop_seconds,
        rereference='average'
    )

    raw = prepare_eeg(
        config,
        BIDS,
        raw=raw,
        apply_sobi=sobi,
        freq_limits=freq_limits_signal,
        metadata_badchannels=True,
        epoch_definition=epoch_definition
    )

    # If sensor
    if 'sensor' in config['feature_extraction']['rel_pow']:

        # Estimate PSD
        adaptive = bool(config['feature_extraction']['rel_pow']['sensor']['adaptive'])
        psd = raw.compute_psd(
            method="multitaper",
            fmin=freq_limits_signal[0],
            fmax=freq_limits_signal[-1],
            bandwidth=config['feature_extraction']['rel_pow']['sensor']['bandwidth'],
            adaptive=adaptive,
            normalization=config['feature_extraction']['rel_pow']['sensor']['normalization']
        )

        # Estimate relative pow
        psds,_ = psd.get_data(return_freqs=True) # (n_channels, n_freqs)
        total_power = numpy.sum(psds, axis=2)  # (n_channels,)
        relative_psd = psds / total_power[:, :, None]

        # Get the metadata to save
        metadata = {
            "method": "multitaper",
            "bandwidth": config['feature_extraction']['rel_pow']['sensor']['bandwidth'],
            "adaptive": config['feature_extraction']['rel_pow']['sensor']['adaptive'],
            "fmin": freq_limits_signal[0],
            "fmax": freq_limits_signal[-1],
            "sfreq": raw.info["sfreq"],
            "epoch_length": raw.tmax - raw.tmin,
            "normalization": "relative_per_epoch_channel",
            "ch_names": raw.ch_names,
            "freqs":psd.freqs,
            "dim": "epochs x sensor x freqs",
            "shape": relative_psd.shape
        }

        # Save
        write_relative_psd(config,BIDS,relative_psd=relative_psd,metadata=metadata)


    return metadata