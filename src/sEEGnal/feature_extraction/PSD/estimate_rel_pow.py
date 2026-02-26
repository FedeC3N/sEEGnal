# -*- coding: utf-8 -*-
"""
Estimate relative power with multitaper for sensors or sources

Federico Ramírez-Toraño
24/02/2026
"""

# Imports
import numpy
import mne

from sEEGnal.tools.mne_tools import prepare_eeg
from sEEGnal.tools.bids_tools import write_relative_psd, read_inverse_solution
from sEEGnal.tools.psd_tools import multitaper_psd, normalize_psd


def estimate_rel_pow(config, BIDS):

    # --------------------------------------------------
    # Load cleaned EEG
    # --------------------------------------------------
    print(f'      Measure: Relative PSD')

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

    raw = prepare_eeg(
        config,
        BIDS,
        preload=True,
        channels_to_include=config['global']["channels_to_include"],
        channels_to_exclude=config['global']["channels_to_exclude"],
        freq_limits=freq_limits_components,
        notch_filter=True,
        resample_frequency=config['component_estimation']['resample_frequency'],
        set_annotations=True,
        crop_seconds=config['component_estimation']['crop_seconds'],
        rereference='average'
    )

    raw = prepare_eeg(
        config,
        BIDS,
        raw=raw,
        apply_sobi=sobi,
        freq_limits=freq_limits_signal,
        metadata_badchannels=True,
        epoch_definition=config['source_reconstruction']['epoch_definition']
    )

    sfreq = raw.info["sfreq"]

    # ==================================================
    # SENSOR LEVEL
    # ==================================================

    if 'sensor' in config['feature_extraction']['rel_pow']:

        print('          Sensors.')

        params = config['feature_extraction']['rel_pow']['sensor']

        # raw is epoched here → shape (n_epochs, n_channels, n_times)
        data = raw.get_data()

        psd, freqs = multitaper_psd(
            data=data,
            sfreq=raw.info['sfreq'],
            fmin=freq_limits_signal[0],
            fmax=freq_limits_signal[-1],
            bandwidth=params['bandwidth'],
            adaptive=params['adaptive'],
            dtype=numpy.float32
        )

        relative_psd = normalize_psd(psd, mode="relative")

        metadata = {
            "method": "custom_multitaper",
            "bandwidth": params['bandwidth'],
            "adaptive": params['adaptive'],
            "fmin": freq_limits_signal[0],
            "fmax": freq_limits_signal[-1],
            "sfreq": sfreq,
            "epoch_length": raw.tmax - raw.tmin,
            "normalization": "relative_per_epoch_channel",
            "ch_names": raw.ch_names,
            "freqs": freqs,
            "dim": "epochs x sensor x freqs",
            "shape": relative_psd.shape
        }

        # Save the result
        config['current_space'] = 'sensor'
        write_relative_psd(
            config,
            BIDS,
            relative_psd=relative_psd,
            metadata=metadata
        )
        del config['current_space']

    # ==================================================
    # SOURCE LEVEL
    # ==================================================

    if 'source' in config['feature_extraction']['rel_pow']:

        print('          Sources.')

        params = config['feature_extraction']['rel_pow']['source']

        dummy = config['subsystem']
        config['subsystem'] = 'source_reconstruction'
        filters = read_inverse_solution(config, BIDS)
        config['subsystem'] = dummy

        # Apply LCMV per epoch
        stcs = mne.beamformer.apply_lcmv_epochs(raw, filters)

        # Stack → (n_epochs, n_vertices, n_times)
        data = numpy.stack([stc.data for stc in stcs])

        psd, freqs = multitaper_psd(
            data=data,
            sfreq=raw.info['sfreq'],
            fmin=freq_limits_signal[0],
            fmax=freq_limits_signal[-1],
            bandwidth=params['bandwidth'],
            adaptive=params['adaptive'],
            dtype=numpy.float32
        )

        # Get relative PSD
        relative_psd = normalize_psd(psd, mode="relative")

        metadata = {
            "method": "custom_multitaper",
            "bandwidth": params['bandwidth'],
            "adaptive": params['adaptive'],
            "fmin": freq_limits_signal[0],
            "fmax": freq_limits_signal[-1],
            "sfreq": sfreq,
            "normalization": "relative_per_epoch_vertex",
            "freqs": freqs,
            "dim": "epochs x vertices x freqs",
            "shape": relative_psd.shape
        }

        # Save the result
        config['current_space'] = 'source'
        write_relative_psd(
            config,
            BIDS,
            relative_psd=relative_psd,
            metadata=metadata
        )
        del config['current_space']

    return metadata