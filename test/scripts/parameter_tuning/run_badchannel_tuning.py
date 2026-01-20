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
import plot_badchannels as pbc
import sEEGnal.preprocess.find_badchannels as fbc
from sEEGnal.tools.bids_tools import create_bids_path


# Parameters
criteria = ['impossible_amplitude_detection', 'component_detection',
            'gel_bridge_detection', 'high_deviation_detection']
criteria = ['all']

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

            func = getattr(fbc, current_criterion)
            current_badchannels = func(config,bids_path)
            badchannels.extend(current_badchannels)

            # Print out information
            print(f"    {current_criterion} bad channels: {current_badchannels}")

        # Plot
        func = getattr(pbc, 'all')
        func(config, bids_path,badchannels)

    else:

        for current_criterion in criteria:
            func = getattr(fbc, current_criterion)
            badchannels = func(config,bids_path)

            # Print out information
            print(f"    {current_criterion} bad channels: {badchannels}")

            # Plot
            func = getattr(pbc, current_criterion)
            func(config, bids_path,badchannels)
