from pathlib import Path
from importlib.resources import files

import json
import numpy

import numpy as np
import pandas as pd
import nibabel as nib


def label_aal(config, src, trans=None):

    # --------------------------------------------------
    # Load atlas (MNI space)
    # --------------------------------------------------
    atlas_path = files("sEEGnal.data") / 'atlas' / 'AAL_MNI_V4.nii'
    labels_path = files("sEEGnal.data") / 'atlas' / 'AAL_MNI_V4.txt'

    atlas_img = nib.load(atlas_path)
    atlas_img = nib.as_closest_canonical(atlas_img)

    atlas_data = atlas_img.get_fdata()
    atlas_affine = atlas_img.affine

    # --------------------------------------------------
    # Load labels (first 90 cortical areas)
    # --------------------------------------------------
    atlas_labels = pd.read_table(labels_path, header=None)
    atlas_labels = atlas_labels[:90]

    relabeled = np.zeros_like(atlas_data)

    for i in range(len(atlas_labels)):
        relabeled[atlas_data == atlas_labels[2][i]] = i + 1

    # --------------------------------------------------
    # Compute centroids in MNI space (meters)
    # --------------------------------------------------
    centroids = np.zeros((len(atlas_labels) + 1, 3))

    for i in range(len(atlas_labels)):
        vox = np.array(np.where(relabeled == i + 1)).T
        if len(vox) == 0:
            continue
        centroid_vox = vox.mean(axis=0)
        centroid_mm = nib.affines.apply_affine(atlas_affine, centroid_vox)
        centroids[i + 1] = centroid_mm * 1e-3  # mm → m

    # --------------------------------------------------
    # Get active source positions (meters)
    # --------------------------------------------------
    src_space = [[i] * s['nuse'] for i, s in enumerate(src)]
    src_space = np.concatenate(src_space)

    src_pos = [s['rr'][s['vertno']] for s in src]
    src_pos = np.concatenate(src_pos)  # shape (N, 3)

    # --------------------------------------------------
    # Apply transformation if provided
    # --------------------------------------------------
    if trans is not None:
        # If trans is an MNE Transform dict
        if isinstance(trans, dict) and 'trans' in trans:
            T = trans['trans']
        else:
            T = trans

        # Apply 4x4 affine to Nx3 coordinates
        src_pos = (
            T[:3, :3] @ src_pos.T + T[:3, 3:4]
        ).T

    # --------------------------------------------------
    # Convert meters → mm (atlas is mm)
    # --------------------------------------------------
    src_pos_mm = src_pos * 1e3

    # --------------------------------------------------
    # Convert MNI mm → atlas voxel indices
    # --------------------------------------------------
    src_vox = nib.affines.apply_affine(
        np.linalg.inv(atlas_affine),
        src_pos_mm
    )

    src_vox = np.round(src_vox).astype(int)

    # --------------------------------------------------
    # Find valid voxel indices
    # --------------------------------------------------
    valid = (
        (src_vox[:, 0] >= 0) & (src_vox[:, 0] < atlas_data.shape[0]) &
        (src_vox[:, 1] >= 0) & (src_vox[:, 1] < atlas_data.shape[1]) &
        (src_vox[:, 2] >= 0) & (src_vox[:, 2] < atlas_data.shape[2])
    )

    src_area = np.zeros(len(src_vox), dtype=int)

    src_area[valid] = relabeled[
        src_vox[valid, 0],
        src_vox[valid, 1],
        src_vox[valid, 2]
    ].astype(int)

    # --------------------------------------------------
    # Build atlas dictionary
    # --------------------------------------------------
    atlas = {
        'atlas_raw': relabeled,
        'label': ['None'] + list(atlas_labels[1]),
        'nick': ['None'] + list(atlas_labels[0]),
        'rr': centroids,
        'src_area': src_area,
        'src_space': src_space.astype(int),
        'src_vox': src_vox
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
