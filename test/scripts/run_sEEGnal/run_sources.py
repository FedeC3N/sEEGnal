#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scripts to call sources_reconstruction

Federico Ramírez-Toraño
11/02/2026

"""

# Imports
from test.init.init import init
from sEEGnal.tools.bids_tools import create_bids_path
from sEEGnal.sources_reconstruction.forward import make_forward_model

# What step to run: forward, inverse
run = [1, 1]

# Init the database
config, files, sub, ses, task = init()

# List of subjects with errors
errors = []

# Go through each subject
for current_index in range(len(files)):

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