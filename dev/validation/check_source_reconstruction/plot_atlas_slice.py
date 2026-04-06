import numpy as np
import matplotlib.pyplot as plt


def plot_area_orthoview(atlas, area_indices, slices, tol=1):
    """
    Plot axial, coronal and sagittal views with multiple atlas areas overlaid.

    Parameters
    ----------
    atlas : dict
        Output from label_aal()
    area_indices : int | list | range
        Atlas region index(es) (1..N)
    slices : tuple of int
        (x_slice, y_slice, z_slice)
    tol : int
        Slice thickness tolerance in voxels
    """

    # --------------------------------------------------
    # Normalize input to list
    # --------------------------------------------------
    if isinstance(area_indices, int):
        area_indices = [area_indices]
    else:
        area_indices = list(area_indices)

    x_slice, y_slice, z_slice = slices

    atlas_data = atlas['atlas_raw']
    src_vox = atlas['src_vox']
    src_area = atlas['src_area']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot anatomical background
    sag = atlas_data[x_slice, :, :]
    cor = atlas_data[:, y_slice, :]
    axi = atlas_data[:, :, z_slice]

    axes[0].imshow(sag.T, origin='lower', cmap='gray')
    axes[1].imshow(cor.T, origin='lower', cmap='gray')
    axes[2].imshow(axi.T, origin='lower', cmap='gray')

    # Generate distinct colors
    cmap = plt.cm.get_cmap("tab10", len(area_indices))

    # --------------------------------------------------
    # Overlay each area
    # --------------------------------------------------
    for idx, area_index in enumerate(area_indices):

        mask = src_area == area_index
        area_vox = src_vox[mask]

        if len(area_vox) == 0:
            continue

        color = cmap(idx)

        # Sagittal
        sag_mask = np.abs(area_vox[:, 0] - x_slice) <= tol
        axes[0].scatter(area_vox[sag_mask, 1],
                        area_vox[sag_mask, 2],
                        s=15, color=color)

        # Coronal
        cor_mask = np.abs(area_vox[:, 1] - y_slice) <= tol
        axes[1].scatter(area_vox[cor_mask, 0],
                        area_vox[cor_mask, 2],
                        s=15, color=color)

        # Axial
        axi_mask = np.abs(area_vox[:, 2] - z_slice) <= tol
        axes[2].scatter(area_vox[axi_mask, 0],
                        area_vox[axi_mask, 1],
                        s=15, color=color)

    # Titles
    axes[0].set_title(f"Sagittal (x={x_slice})")
    axes[1].set_title(f"Coronal (y={y_slice})")
    axes[2].set_title(f"Axial (z={z_slice})")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    labels = [atlas['label'][i] for i in area_indices]
    fig.suptitle(" | ".join(labels))

    plt.tight_layout()
    plt.show()
