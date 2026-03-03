from pathlib import Path
from importlib.resources import files

import mne
import json
import numpy

import numpy as np
import pandas as pd
import nibabel as nib


def label_aal(config, src, trans=None):
    # --------------------------------------------------
    # Locate fsaverage label directory
    # --------------------------------------------------
    subjects_dir = files("sEEGnal.data")
    subjects_dir = Path(subjects_dir)

    subject = config['source_reconstruction']['forward']['template']['subject']

    # --------------------------------------------------
    # Read annot files
    # --------------------------------------------------
    labels_lh = mne.read_labels_from_annot(
        subject=subject,
        parc="AAL",
        hemi="lh",
        subjects_dir=subjects_dir
    )

    labels_rh = mne.read_labels_from_annot(
        subject=subject,
        parc="AAL",
        hemi="rh",
        subjects_dir=subjects_dir
    )

    labels_all = labels_lh + labels_rh

    # --------------------------------------------------
    # Prepare vertex indexing
    # --------------------------------------------------
    n_vertices = sum(len(s['vertno']) for s in src)

    src_area = np.zeros(n_vertices, dtype=int)
    src_space = []
    rr = []

    offset = 0
    label_names = ["None"]

    # --------------------------------------------------
    # Assign labels hemisphere by hemisphere
    # --------------------------------------------------
    for hemi_idx, (s, hemi_labels) in enumerate(zip(src, [labels_lh, labels_rh])):

        vertno = s['vertno']
        rr.append(s['rr'][vertno])

        # Track hemisphere index per vertex
        src_space.append(np.full(len(vertno), hemi_idx, dtype=int))

        # Assign labels
        for label_id, label in enumerate(hemi_labels, start=1):
            # Global label index (avoid overlap between hemispheres)
            global_label_id = len(label_names)

            mask = np.isin(vertno, label.vertices)
            src_area[offset:offset + len(vertno)][mask] = global_label_id

            label_names.append(label.name)

        offset += len(vertno)

    # Concatenate hemisphere info
    src_space = np.concatenate(src_space)
    rr = np.concatenate(rr)

    # --------------------------------------------------
    # Build atlas dictionary
    # --------------------------------------------------
    atlas = {
        'label': label_names,
        'src_area': src_area,  # label index per vertex
        'src_space': src_space,  # hemisphere index per vertex
        'rr': rr  # vertex coordinates (meters)
    }

    return atlas



def label_rsn(config,src,trans=None):
    # --------------------------------------------------
    # Locate fsaverage label directory
    # --------------------------------------------------
    subjects_dir = files("sEEGnal.data")
    subjects_dir = Path(subjects_dir)

    subject = config['source_reconstruction']['forward']['template']['subject']

    # --------------------------------------------------
    # Read annot files
    # --------------------------------------------------
    labels_lh = mne.read_labels_from_annot(
        subject=subject,
        parc="atlas55_RSN",
        hemi="lh",
        subjects_dir=subjects_dir
    )

    labels_rh = mne.read_labels_from_annot(
        subject=subject,
        parc="atlas55_RSN",
        hemi="rh",
        subjects_dir=subjects_dir
    )

    labels_all = labels_lh + labels_rh

    # --------------------------------------------------
    # Prepare vertex indexing
    # --------------------------------------------------
    n_vertices = sum(len(s['vertno']) for s in src)

    src_area = np.zeros(n_vertices, dtype=int)
    src_space = []
    rr = []

    offset = 0
    label_names = ["None"]

    # --------------------------------------------------
    # Assign labels hemisphere by hemisphere
    # --------------------------------------------------
    for hemi_idx, (s, hemi_labels) in enumerate(zip(src, [labels_lh, labels_rh])):

        vertno = s['vertno']
        rr.append(s['rr'][vertno])

        # Track hemisphere index per vertex
        src_space.append(np.full(len(vertno), hemi_idx, dtype=int))

        # Assign labels
        for label_id, label in enumerate(hemi_labels, start=1):
            # Global label index (avoid overlap between hemispheres)
            global_label_id = len(label_names)

            mask = np.isin(vertno, label.vertices)
            src_area[offset:offset + len(vertno)][mask] = global_label_id

            label_names.append(label.name)

        offset += len(vertno)

    # Concatenate hemisphere info
    src_space = np.concatenate(src_space)
    rr = np.concatenate(rr)

    # --------------------------------------------------
    # Build atlas dictionary
    # --------------------------------------------------
    atlas = {
        'label': label_names,
        'src_area': src_area,  # label index per vertex
        'src_space': src_space,  # hemisphere index per vertex
        'rr': rr  # vertex coordinates (meters)
    }

    return atlas