#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to test one by one the bad channel detection and artifact detection
criteria

Federico Ramírez-Toraño
19/01/2026

"""

# Imports
import numpy as np

from test.init.init import init
import plot_artifacts as par
import sEEGnal.preprocess.artifact_detection as far
from sEEGnal.tools.bids_tools import create_bids_path, write_annotations


# Parameters
criteria = ['EOG_detection', 'muscle_detection',
            'sensor_detection', 'other_detection']
criteria = ['sensor_detection']

# Init the database
config, files, sub, ses, task = init()

# Random order to avoid seen the same subjects over and over again
permutated_index = np.random.permutation(len(files))

for subject_index in permutated_index:

    # Get current info
    current_file = files[subject_index]
    current_sub = sub[subject_index]
    current_ses = ses[subject_index]
    current_task = task[subject_index]

    print(f"{subject_index} - {current_file}")

    # Create the subjects following AI-Mind protocol
    bids_path = create_bids_path(config, current_sub, current_ses, current_task)

    if criteria == ['all']:

        dummy_criteria = ['impossible_amplitude_detection', 'component_detection',
                'gel_bridge_detection', 'high_deviation_detection']
        badchannels = []

        for current_criterion in dummy_criteria:

            func = getattr(far, current_criterion)
            current_badchannels = func(config,bids_path)
            badchannels.extend(current_badchannels)

            # Print out information
            print(f"    {current_criterion} bad channels: {current_badchannels}")

        # Plot
        func = getattr(par, 'all')
        func(config, bids_path,badchannels)

    else:

        for current_criterion in criteria:

            # CAll the function
            func = getattr(far, current_criterion)
            if current_criterion == 'muscle_detection' or current_criterion == 'sensor_detection':
                annotations = func(config, bids_path,'sobi')
            else:
                annotations = func(config,bids_path)

            # Save the annotations in BIDS format
            _ = write_annotations(bids_path, annotations)

            # Print out information
            print(f"    {current_criterion}: {annotations}")

            # Plot
            func = getattr(par, current_criterion)
            func(config, bids_path)
