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


def save_atlases(
        filename,
        atlases):
    # Makes a copy of the atlases.
    atlases = atlases.copy()

    # Goes through each atlas.
    for atlas in atlases:
        # Converts the area and sources space labels into lists of integers.
        atlases[atlas]['src_area'] = [int(label) for label in atlases[atlas]['src_area']]
        atlases[atlas]['src_space'] = [int(label) for label in atlases[atlas]['src_space']]

        # Converts the atlas centroids into lists of lists.
        atlases[atlas]['rr'] = [[float(coord) for coord in centroid] for centroid in atlases[atlas]['rr']]

    # Saves the atlas as a JSON file.
    with open(filename, 'w') as fid:
        json.dump(atlases, fid, indent=2)


def load_atlases(
        filename):
    # Loads the JSON file as a dictionary.
    with open(filename, 'r') as fid:
        atlases = json.load(fid)

    # Goes through each atlas.
    for atlas in atlases:
        # Converts the area and sources space labels into numpy arrays.
        atlases[atlas]['src_area'] = numpy.array(atlases[atlas]['src_area'])
        atlases[atlas]['src_space'] = numpy.array(atlases[atlas]['src_space'])

        # Converts the atlas centroids into numpy arrays.
        atlases[atlas]['rr'] = numpy.array(atlases[atlas]['rr'])

    # Returns the atlas definition.
    return atlases


def save_src_atlas(
        filename,
        atlas):
    # Makes a copy of the atlas.
    atlas = atlas.copy()

    # Converts the area and sources space labels into lists of integers.
    atlas['src_area'] = [int(label) for label in atlas['src_area']]
    atlas['src_space'] = [int(label) for label in atlas['src_space']]

    # Converts the atlas centroids into lists of lists.
    atlas['rr'] = [[float(coord) for coord in centroid] for centroid in atlas['rr']]

    # Saves the atlas as a JSON file.
    with open(filename, 'w') as fid:
        json.dump(atlas, fid, indent=2)


def load_src_atlas(
        filename):
    # Loads the JSON file as a dictionary.
    with open(filename, 'r') as fid:
        atlas = json.load(fid)

    # Converts the area and sources space labels into numpy arrays.
    atlas['src_area'] = numpy.array(atlas['src_area'])
    atlas['src_space'] = numpy.array(atlas['src_space'])

    # Converts the atlas centroids into numpy arrays.
    atlas['rr'] = numpy.array(atlas['rr'])

    # Returns the atlas definition.
    return atlas
