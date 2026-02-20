#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot forward model

Federico Ramírez-Toraño
12/02/2026

"""


# Imports
import copy
import os
import importlib
from pathlib import Path

import mne
import numpy as np
from mne.transforms import Transform

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
    with importlib.resources.as_file(pkg_template) as subjects_dir:
        subjects_dir = Path(subjects_dir)
        subject = config['source_reconstruction']['forward']['template']['subject']

        # Copy src
        src_mri = copy.deepcopy(forward_model['src'])

        # Create inverse Transform from mri_head_t
        mri_head_t = forward_model['mri_head_t']  # this is an MNE Transform
        head_mri_t = mne.transforms.invert_transform(mri_head_t)

        # Apply inverse to each source point
        for s in src_mri:
            if s['type'] == 'vol':
                coords_hom = np.hstack([s['rr'], np.ones((s['rr'].shape[0], 1))])
                coords_mri = (head_mri_t['trans'] @ coords_hom.T).T[:, :3]
                s['rr'] = coords_mri

        # Plot
        for view in ['coronal', 'axial', 'sagittal']:

            fig = mne.viz.plot_bem(
                src=src_mri,
                trans=Transform('mri', 'head', np.eye(4)),  # identity transform
                subject=subject,
                subjects_dir=subjects_dir,
                brain_surfaces='white',
                orientation=view,
                slices=[50, 100, 150, 200],
                show=False
            )

            # Save the figure
            process = 'check'
            tail = f"sources_pos_{view}"
            figure_path = build_derivatives_path(BIDS, process, tail)
            if not (os.path.exists(figure_path.parent)):
                os.makedirs(figure_path.parent)
            plt.savefig(figure_path)
            plt.close()







