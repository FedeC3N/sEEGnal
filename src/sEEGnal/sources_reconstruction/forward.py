#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the forward model

Federico Ramírez-Toraño
11/02/2026

"""

# Import
import traceback
from pathlib import Path
from importlib.resources import files, as_file
from datetime import datetime as dt, timezone

import mne
from mne.transforms import Transform

from sEEGnal.tools.mne_tools import prepare_eeg
from sEEGnal.tools.bids_tools import write_forward_model


def make_forward_model(config, BIDS):

    # Add the subsystem info
    config['subsystem'] = 'source_reconstruction'

    try:

        # Choose the correct estimation
        if config['source_reconstruction']['forward']['use_template']:

            # Estimate the forward model
            forward_model = template_forward_model(config, BIDS)

        else:
            forward_model = []


        # Save the metadata
        now = dt.now(timezone.utc)
        formatted_now = now.strftime("%d-%m-%Y %H:%M:%S")
        results = {
            'result': 'ok',
            'bids_basename': BIDS.basename,
            "date": formatted_now,
            'forward_model': forward_model
        }

    except Exception as e:

        # Save the error
        now = dt.now(timezone.utc)
        formatted_now = now.strftime("%d-%m-%Y %H:%M:%S")
        results = {
            'result': 'error',
            'bids_basename': BIDS.basename,
            "date": formatted_now,
            "details": f"Exception: {str(e)}, {traceback.format_exc()}"
        }



    return results


def template_forward_model(config, BIDS):
    """

    Use the Freesurfer template to create a forward model

    """

    # Load the EEG to obtain the montage
    raw = prepare_eeg(
        config,
        BIDS
    )

    # Get the FreeSurfer fsaverage information
    pkg_fsaverage = files("sEEGnal.data")
    with as_file(pkg_fsaverage) as subjects_dir:
        subjects_dir = Path(subjects_dir)
        subject = config['source_reconstruction']['forward']['template']['subject']
        bem = subjects_dir / subject / 'bem' / config['source_reconstruction']['forward']['template']['bem']
        trans = config['source_reconstruction']['forward']['template']['trans']
        surf = subjects_dir / subject / 'bem' / config['source_reconstruction']['forward']['template']['surf']

        # Define our sources
        spacing = config['source_reconstruction']['forward']['template']['spacing']
        src = mne.setup_source_space(
            subject=subject,
            spacing=spacing,
            subjects_dir=subjects_dir,
            add_dist=False
        )

        # Read the BEM solution (MNI template)
        bem = mne.read_bem_solution(bem)

        # Create the output forward model
        forward_model = mne.make_forward_solution(
            raw.info,
            trans,
            src,
            bem,
            meg=False,
            eeg=True
        )

    # Save the forward solution
    write_forward_model(config, BIDS, forward_model)

    # Save the forward model
    return forward_model