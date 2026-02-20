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
from mne.transforms import Transform

import numpy
import nibabel
import matplotlib
matplotlib.use('Qt5Agg')
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
    src = forward_model['src']

    # Load the template
    pkg_template = importlib.resources.files("sEEGnal.data")
    with importlib.resources.as_file(pkg_template) as fs_dir:
        fs_dir = Path(fs_dir)
        subject = config['source_reconstruction']['forward']['template']['subject']
        mri_path = fs_dir / subject / "mri" / config['source_reconstruction']['forward']['template']['mri']
        mri = nibabel.load(mri_path)
        mri = nibabel.as_closest_canonical(mri)

        # Move axes content
        for view in ['coronal', 'axial', 'sagittal']:

            trans = Transform('head', 'mri')
            plot_bem_kwargs = dict(
                subject=subject,
                subjects_dir=fs_dir,
                brain_surfaces="white",
                orientation=view,
                slices=[50, 100, 150, 200],
                show=False
            )
            fig = mne.viz.plot_bem(src=forward_model['src'], trans=trans, **plot_bem_kwargs)

            # Save the figure
            process = 'check'
            tail = f"sources_pos_{view}"
            figure_path = build_derivatives_path(BIDS, process, tail)
            if not (os.path.exists(figure_path.parent)):
                os.makedirs(figure_path.parent)
            plt.savefig(figure_path)
            plt.close()







