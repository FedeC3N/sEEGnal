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
from sEEGnal.tools.bids_tools import read_inverse_solution, write_plv
from sEEGnal.tools.fc_tools import compute_plv


def estimate_plv(config, BIDS):

    # --------------------------------------------------
    # Load cleaned EEG
    # --------------------------------------------------
    print(f'      Measure: PLV')

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
        config['feature_extraction']['plv']['freq_limits'][0],
        config['feature_extraction']['plv']['freq_limits'][-1]
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
        epoch_definition=config['feature_extraction']['plv']['epoch_definition']
    )

    # SENSOR LEVEL
    if 'sensor' in config['feature_extraction']['plv']:

        print('          Sensors.')

        params = config['feature_extraction']['plv']['sensor']

        # Save memory
        nepochs, nsources, nsamples = raw.get_data().shape
        nbands = len(params['freq_bands_name'])
        conn_indices = numpy.triu_indices(nsources, k=1)
        n_connections = len(conn_indices[0])
        plv = numpy.zeros((nepochs, nbands, n_connections), dtype=numpy.float32)

        for iband,current_band in enumerate(params['freq_bands_name']):

            # Filter the data
            banddata = raw.copy().load_data()
            banddata.filter(
                params['freq_bands_limits'][iband][0],
                params['freq_bands_limits'][iband][1]
            )

            # PLV
            band_plv_vector = compute_plv(
                data=banddata.get_data()
            )

            # Save it
            plv[:,iband,:] = band_plv_vector

        metadata = {
            "method": "PLV",
            "freq_bands_name": params['freq_bands_name'],
            "freq_bands_limits":  params['freq_bands_limits'],
            "epoch_length": raw.tmax - raw.tmin,
            "ch_names": raw.ch_names,
            "dim": "epochs × bands × connections",
            "shape": plv.shape,
            "triu_indices": conn_indices
        }

        # Save the result
        config['current_space'] = 'sensor'
        write_plv(
            config,
            BIDS,
            plv=plv,
            metadata=metadata
        )
        del config['current_space']

    # SOURCE LEVEL
    if 'source' in config['feature_extraction']['plv']:

        print('          Sources.')

        params = config['feature_extraction']['plv']['source']

        dummy = config['subsystem']
        config['subsystem'] = 'source_reconstruction'
        filters = read_inverse_solution(config, BIDS)
        config['subsystem'] = dummy

        # Apply LCMV per epoch
        stcs = mne.beamformer.apply_lcmv_epochs(raw, filters)


        # Save memory
        nepochs = len(stcs)
        nsources, nsamples = stcs[0].shape
        nbands = len(params['freq_bands_name'])
        conn_indices = numpy.triu_indices(nsources, k=1)
        n_connections = len(conn_indices[0])
        plv = numpy.zeros((nepochs, nbands, n_connections), dtype=numpy.float32)

        for iband, current_band in enumerate(params['freq_bands_name']):

            # Filter the data
            filtered_stcs = []
            for stc in stcs:
                stc_copy = stc.copy()
                stc_copy.filter(
                    params['freq_bands_limits'][iband][0],
                    params['freq_bands_limits'][iband][1]
                )
                filtered_stcs.append(stc_copy)

            banddata = numpy.stack([stc.data for stc in filtered_stcs])

            # PLV
            band_plv_vector = compute_plv(
                data=banddata
            )

            # Save it
            plv[:, iband, :] = band_plv_vector

        metadata = {
            "method": "PLV",
            "freq_bands_name": params['freq_bands_name'],
            "freq_bands_limits": params['freq_bands_limits'],
            "epoch_length": raw.tmax - raw.tmin,
            "dim": "epochs × bands × connections",
            "shape": plv.shape,
            "triu_indices": "numpy.triu_indices(nsources, k=1)"
        }

        # Save the result
        config['current_space'] = 'source'
        write_plv(
            config,
            BIDS,
            plv=plv,
            metadata=metadata
        )
        del config['current_space']


    metadata = []
    return metadata