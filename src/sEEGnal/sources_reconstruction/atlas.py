import importlib.resources
import json

import numpy
import pandas

import nibabel, nibabel.affines

# Sets the sources data folder.
metabase = (
    importlib.resources.files('aimind.meeg').
    joinpath('sources').
    joinpath('data'))


def label_aal(src):
    # Defines the base folder for the atlas data.
    base = metabase.joinpath('atlas')

    # Loads the AAL atlas.
    atlas_mri = nibabel.load(base.joinpath('ROI_MNI_V4.nii'))
    atlas_lab = pandas.read_table(str(base.joinpath('ROI_MNI_V4.txt')), header=None)

    # Removes the non-cortical areas.
    atlas_lab = atlas_lab[:90]

    # Initializes the raw 3D array for the relabeled atlas matrix.
    atlas_raw = numpy.zeros(atlas_mri.shape)

    # Relabels the atlas areas as 1 to N.
    for i in range(len(atlas_lab)):
        # Looks for the voxels with the current label.
        hits = atlas_mri.get_fdata() == atlas_lab[2][i]

        # Re-labels the voxels.
        atlas_raw[hits] = i + 1

    # Initializes the list of area centroids.
    centroids = numpy.zeros([len(atlas_lab) + 1, 3])

    # Gets the centroid for each area.
    for i in range(len(atlas_lab)):
        # Looks for all the voxels in the current area.
        voxels = numpy.where(atlas_raw == i + 1)
        centroid = numpy.mean(voxels, axis=1)

        # Transforms the centroid from voxel to RAS (MNI space) coordinates.
        centroid = nibabel.affines.apply_affine(atlas_mri.affine, centroid)

        # Transforms the centroid into SI units (meters).
        centroid = centroid * 1e-3

        # Stores the centroid.
        centroids[i + 1] = centroid

    # Transforms the volumetric grid into MRI voxels.

    # Concatenates the sources in use in all the source spaces.
    src_space = [[i] * s['nuse'] for i, s in enumerate(src)]
    src_space = numpy.concatenate(src_space)
    src_pos = [s['rr'][s['vertno']] for s in src]
    src_pos = numpy.concatenate(src_pos)

    # Transforms the source positions into millimeters.
    src_pos = src_pos * 1e3

    # Applies the affine transformation from RAS (MNI space) to voxels.
    src_vox = nibabel.affines.apply_affine(numpy.linalg.inv(atlas_mri.affine), src_pos)
    src_vox = src_vox.round().astype(int)

    # Initializes the list of source areas.
    src_area = numpy.zeros(src_vox.shape[0])

    # Lists the valid sources (those falling inside the atlas MRI space).
    valid = numpy.all(src_vox >= 0, axis=-1) & numpy.all(src_vox < atlas_mri.shape, axis=-1)

    # Transforms the 3D voxel subindexes into 1D flat indexes of the atlas.
    src_ind = numpy.ravel_multi_index(tuple(src_vox[valid].T), atlas_mri.shape)

    # Gets the atlas index for each (valid) source in the source model.
    src_area[valid] = atlas_raw.flat[src_ind].astype(int)

    # Creates the atlas dictionary.
    atlas = {
        'label': ['None'] + list(atlas_lab[1]),
        'nick': ['None'] + list(atlas_lab[0]),
        'rr': centroids,
        'src_area': src_area.astype(int),
        'src_space': src_space.astype(int)}

    # Returns the atlas definition for these source spaces.
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
