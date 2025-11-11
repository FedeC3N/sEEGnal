#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to call automatic preprocess for all EEGs

Federico Ramírez-Toraño
28/10/2025

"""

# Imports
from init import init
from sEEGnal.tools.bids_tools import create_bids_path
from sEEGnal.standardize.standardize import standardize
from sEEGnal.preprocess.artifact_detection import artifact_detection
from sEEGnal.preprocess.badchannel_detection import badchannel_detection


# What step to run: standardize, badchannel, artifact
run = [0,1,1]

# Init the database
config, files, sub, ses, task = init()

# List of subjects with errors
errors = []

# Go through each subject
for current_index in range(len(files)):

        # current info
        current_file    = files[current_index]
        current_sub     = sub[current_index]
        current_ses     = ses[current_index]
        current_task    = task[current_index]

        # Create the subjects following AI-Mind protocol
        bids_path = create_bids_path(config,current_file,current_sub,current_ses,current_task)

        print('Working with sub ' + current_sub + ' ses ' + current_ses + ' task ' + current_task)

        # Run the selected processes
        if run[0]:
            print('   Standardize', end='. ')
            results = standardize(config,current_file,bids_path)
            print(' Result ' + results['result'])

        if run[1]:
            print('   Badchannel detection', end='. ')
            results = badchannel_detection(config,bids_path)
            print(' Result ' + results['result'])
            if results['result'] == 'error' and config['global']['verbose'] == 'full':
                print(results['details'])

        if run[2]:
            print('   Artifact Detection', end='. ')
            results = artifact_detection(config, bids_path)
            print(' Result ' + results['result'])
            if results['result'] == 'error' and config['global']['verbose'] == 'full':
                print(results['details'])


        print()
        print()
        print()


