#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to test one by one the bad channel detection and artifact detection
criteria

Federico Ramírez-Toraño
19/01/2026

"""

# Imports
from test.init.init import init
import sEEGnal.preprocess.find_badchannels as fbc
import plot_badchannels as pbc
from sEEGnal.tools.bids_tools import create_bids_path


# Parameters
criteria = ['impossible_amplitude_detection', 'power_spectrum_detection',
            'gel_bridge_detection', 'high_deviation_detection']
criteria = ['power_spectrum_detection']

# Init the database
config, files, sub, ses, task = init()


for subject_index in range(len(files)):

    # Get current info
    current_file = files[subject_index]
    current_sub = sub[subject_index]
    current_ses = ses[subject_index]
    current_task = task[subject_index]

    print(f"{subject_index} - {current_file}")

    # Create the subjects following AI-Mind protocol
    bids_path = create_bids_path(config, current_sub, current_ses, current_task)

    for current_criterion in criteria:

        # Call the specific function
        func = getattr(fbc, current_criterion)
        badchannels = func(config,bids_path)

        # Print out information
        print(f"    {current_criterion} bad channels: {badchannels}")

        # Plot
        func = getattr(pbc, current_criterion)
        func(config, bids_path,badchannels)
