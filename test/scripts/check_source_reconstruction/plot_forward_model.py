#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot forward model

Federico Ramírez-Toraño
12/02/2026

"""


# Imports
import os
from pathlib import Path
import importlib

import mne
import pyvista as pv
import matplotlib.pyplot as plt

from test.init.init import init
from sEEGnal.tools.bids_tools import build_BIDS_object, read_forward_model, build_derivatives_path

# Init the database
config, files, sub, ses, task = init()


# Go through each subject
index = range(len(files))
for current_index in index:

    # current info
    current_file = files[current_index]
    current_sub = sub[current_index]
    current_ses = ses[current_index]
    current_task = task[current_index]

    print(current_file)

    # Create the subjects following AI-Mind protocol
    BIDS = build_BIDS_object(config, current_sub, current_ses, current_task)

    # Read forward model
    config['subsystem'] = 'source_reconstruction'
    forward_model = read_forward_model(config,BIDS)
    src = forward_model['src'][0]

    # Get the FreeSurfer fsaverage information
    pkg_fsaverage = importlib.resources.files("sEEGnal.data")
    with (importlib.resources.as_file(pkg_fsaverage) as fs_dir):


        fs_dir = Path(fs_dir)
        subject = config['source_reconstruction']['forward']['template']['subject']
        trans = mne.transforms.Transform(
            forward_model['mri_head_t']['from'],
            forward_model['mri_head_t']['to'],
            trans=forward_model['mri_head_t']['trans']
        )

        pv.OFF_SCREEN = True  # Windows compatible
        fig = mne.viz.plot_alignment(
            info=forward_model['info'],
            trans=trans,
            subject=subject,
            subjects_dir=fs_dir,
            surfaces='white',
            src=forward_model['src'],
            coord_frame='head',
            eeg=True,
            meg=False
        )

        # Save the figure
        process = 'check'
        tail = 'sources_alignment'
        figure_path = build_derivatives_path(BIDS, process, tail)
        if not (os.path.exists(figure_path.parent)):
            os.makedirs(figure_path.parent)
        screenshot = fig.plotter.screenshot(figure_path)
        fig.plotter.close()







