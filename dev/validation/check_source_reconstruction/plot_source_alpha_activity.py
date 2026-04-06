"""
Topoplot alpha activity

Federico Ramírez-Toraño
17/02/2026

"""

# Imports
import os
import copy
import importlib
from pathlib import Path

import mne
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from scipy.signal import welch

from dev.init.init import init
from sEEGnal.tools.mne_tools import prepare_eeg
from sEEGnal.tools.bids_tools import build_BIDS_object, read_inverse_solution, read_forward_model, build_derivatives_path
from sEEGnal.tools.template_tools import get_subjects_dir


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

    # Add the subsystem info
    config['subsystem'] = 'source_reconstruction'

    # Load the clean EEG
    sobi = {
        'desc': 'sobi',
        'components_to_include': ['brain', 'other'],
        'components_to_exclude': []
    }
    freq_limits = [
        config['component_estimation']['low_freq'],
        config['component_estimation']['high_freq']
    ]
    crop_seconds = config['component_estimation']['crop_seconds']
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]
    epoch_definition = config['source_reconstruction']['epoch_definition']

    # Load the clean data
    raw = prepare_eeg(
        config,
        BIDS,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        freq_limits=freq_limits,
        notch_filter=True,
        resample_frequency=resample_frequency,
        set_annotations=True,
        crop_seconds=crop_seconds,
        rereference='average'
    )

    raw = prepare_eeg(
        config,
        BIDS,
        raw=raw,
        apply_sobi=sobi,
        freq_limits=[2, 45],
        metadata_badchannels=True,
        epoch_definition=epoch_definition
    )

    # Get the atlas-sources information
    forward_model = read_forward_model(config,BIDS)
    mri_head_t = forward_model['mri_head_t']
    head_mri_t = mne.transforms.invert_transform(mri_head_t)

    # Read LCMV beamformer
    lcmv_filters = read_inverse_solution(config,BIDS)

    # Apply filters
    stc = mne.beamformer.apply_lcmv_epochs(raw, lcmv_filters)

    # Compute PSD
    sfreq = raw.info['sfreq']
    alpha_pow = np.zeros((forward_model['nsource'],len(stc)))

    for iepoch,ts in enumerate(stc):
        f, pow = welch(ts.data,
                       fs=sfreq,
                       nperseg=sfreq * 2,
                       scaling='spectrum'
                       )

        # Get the alpha band power
        f_mask = (f > 8) & (f < 12)
        current_alpha_pow = np.sum(pow[:, f_mask], axis=1)

        # Store it
        alpha_pow[:, iepoch] = current_alpha_pow

    # Average across epochs
    alpha_pow = np.mean(alpha_pow,axis=1)

    # Copy src
    src_mri = copy.deepcopy(forward_model['src'])

    # Get only the vertices actually used in the source space
    rr_lh = src_mri[0]['rr'][src_mri[0]['vertno']]
    rr_rh = src_mri[1]['rr'][src_mri[1]['vertno']]
    rr_used = np.vstack([rr_lh, rr_rh])

    # Transform source coordinates from head to MRI space
    coords_hom = np.hstack([rr_used, np.ones((rr_used.shape[0], 1))])
    sources_mri_space = (head_mri_t['trans'] @ coords_hom.T).T[:, :3]

    # Load the template
    subject = config['source_reconstruction']['forward']['template']['subject']
    subjects_dir, subject = get_subjects_dir(subject)

    mri_path = subjects_dir / subject / 'mri' / 'brain.mgz'
    mri_img = nib.load(mri_path)

    mri_data = mri_img.get_fdata()
    affine = mri_img.header.get_vox2ras_tkr()

    # Convert to voxel
    sources_mri_space = sources_mri_space * 1e3
    affine_inv = np.linalg.inv(affine)
    sources_hom = np.hstack([sources_mri_space,
                             np.ones((sources_mri_space.shape[0], 1))])
    vox_coords = (affine_inv @ sources_hom.T).T[:, :3]
    vox_coords = np.round(vox_coords).astype(int)

    # Get the slices to plot
    tol = 3

    # Coronal (y fijo)
    unique_y = np.sort(np.unique(vox_coords[:, 1]))
    slice_y = unique_y[len(unique_y) // 2]
    mask_y = np.abs(vox_coords[:, 1] - slice_y) < tol
    cor_sources = vox_coords[mask_y]
    cor_values = alpha_pow[mask_y]

    # Axial (z fijo)
    unique_z = np.sort(np.unique(vox_coords[:, 2]))
    slice_z = unique_z[len(unique_z) // 2]
    mask_z = np.abs(vox_coords[:, 2] - slice_z) < tol
    axi_sources = vox_coords[mask_z]
    axi_values = alpha_pow[mask_z]

    fig, axs = plt.subplots(1, 2, figsize=(10, 10))

    # Plot
    # Axial
    axs[0].imshow(
        mri_data[:, slice_y, :].T,
        origin='lower',
        cmap='gray'
    )

    axs[0].scatter(
        cor_sources[:, 0],  # x
        cor_sources[:, 2],  # z
        c=cor_values,
        cmap='hot',
        s=20
    )

    axs[0].set_title("Axial")

    # Coronal
    axs[1].imshow(
        mri_data[:, :, slice_z].T,
        origin='lower',
        cmap='gray'
    )

    axs[1].scatter(
        axi_sources[:, 0],  # x
        axi_sources[:, 1],  # y
        c=axi_values,
        cmap='hot',
        s=20
    )

    axs[1].set_title("Coronal")

    # Save output
    plt.tight_layout()
    process = 'check'
    tail = "sources_alpha_activation"
    figure_path = build_derivatives_path(BIDS, process, tail)
    if not (os.path.exists(figure_path.parent)):
        os.makedirs(figure_path.parent)
    plt.savefig(figure_path)
    plt.close()
