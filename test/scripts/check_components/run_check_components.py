#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to plot IC time series and their power spectrum

Federico Ramírez-Toraño
04/02/2026

"""

# Imports
import numpy as np

from test.init.init import init
from sEEGnal.tools.bids_tools import create_bids_path, read_sobi
from test.scripts.check_components.plot_power import plot_component_power

# Init the database
config, files, sub, ses, task = init()

# Random order to avoid seen the same subjects over and over again
index = range(len(files))
index = np.random.permutation(len(files))
for subject_index in index:

    current_file = files[subject_index]
    current_sub = sub[subject_index]
    current_ses = ses[subject_index]
    current_task = task[subject_index]

    # Create the subjects following AI-Mind protocol
    bids_path = create_bids_path(config, current_sub, current_ses, current_task)

    print(bids_path.basename)

    # Plots
    plot_component_power(config,bids_path)

