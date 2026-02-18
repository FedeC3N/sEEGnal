#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topoplot alpha activity

Federico Ramírez-Toraño
17/02/2026

"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot forward model

Federico Ramírez-Toraño
12/02/2026

"""


# Imports
import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import mne
from test.init.init import init

# Init the database
config, _, _, _, _ = init()

# Get the FreeSurfer fsaverage information
fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
subject = config['source_reconstruction']['forward']['template']['subject']
trans = config['source_reconstruction']['forward']['template']['trans']
bem = fs_dir / "bem" / config['source_reconstruction']['forward']['template']['bem']

# fsaverage T1 MRI
mri_file = os.path.join(fs_dir, 'mri', 'T1.mgz')

# Load MRI
mri = nib.load(mri_file)
mri_data = mri.get_fdata()

# Define our sources
src = mne.setup_volume_source_space(
    subject=subject,
    pos=config['source_reconstruction']['forward']['template']['pos'],
    mri=config['source_reconstruction']['forward']['template']['mri'],
    bem=None,
    add_interpolator=True
)

# Get the positions in mm and then voxels
coords = src[0]['rr']
coords_vox = nib.affines.apply_affine(np.linalg.inv(mri.affine), coords * 1000)  # metros -> mm
coords_vox = np.round(coords_vox).astype(int)

# Filter to plot only sources of interest
forward_model =

# Plot
slices = np.unique(coords_vox[:,1])
index = range(0,len(slices),4)
slices = slices[index]

fig, axes = plt.subplots(1, len(slices), figsize=(15,4))
for i, y in enumerate(slices):
    ax = axes[i]
    slice_data = mri_data[:, y, :].T
    ax.imshow(slice_data, cmap='gray', origin='lower')
    mask_slice = coords_vox[:,1] == y
    ax.scatter(coords_vox[mask_slice,0], coords_vox[mask_slice,2], s=5,
               color='red',alpha=0.6)
    ax.set_title(f'Coronal y={y}')
    ax.axis('off')

plt.tight_layout()
plt.show(block=True)
