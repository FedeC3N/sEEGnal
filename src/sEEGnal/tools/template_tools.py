from pathlib import Path
from importlib.resources import files, as_file
import shutil

import mne
import numpy as np


def get_subjects_dir(subject: str) -> tuple[Path, str]:
    """
    Ensure that the fsaverage template and custom annotation files required by
    sEEGnal are available locally.

    Parameters
    ----------
    subject : str
        Template subject name. Currently only "fsaverage" is supported.

    Returns
    -------
    subjects_dir : pathlib.Path
        Path to the FreeSurfer subjects directory containing fsaverage.
    subject : str
        Normalized subject name ("fsaverage").

    Notes
    -----
    This function:
    1. Ensures that MNE's fsaverage dataset is available locally.
    2. Ensures that all custom .annot files packaged with sEEGnal are copied
       into fsaverage/label if they are missing.
    """

    if subject != "fsaverage":
        raise ValueError(
            f"Unsupported template subject: {subject}. "
            "Currently only 'fsaverage' is supported."
        )

    # Fetch or validate fsaverage from MNE
    fsaverage_dir = Path(mne.datasets.fetch_fsaverage(verbose=False))
    subjects_dir = fsaverage_dir.parent
    subject = "fsaverage"

    # Destination label directory inside MNE fsaverage
    label_dir = subjects_dir / subject / "label"
    label_dir.mkdir(parents=True, exist_ok=True)

    # Locate packaged custom annotations
    annots_resource = files("sEEGnal.data").joinpath("annots")

    with as_file(annots_resource) as annots_path:
        annots_path = Path(annots_path)

        if not annots_path.exists():
            raise FileNotFoundError(
                f"Custom annotation directory not found: {annots_path}"
            )

        annot_files = sorted(annots_path.glob("*.annot"))

        if not annot_files:
            raise FileNotFoundError(
                f"No custom .annot files found in: {annots_path}"
            )

        for annot_file in annot_files:
            dst = label_dir / annot_file.name
            if not dst.exists():
                shutil.copy2(annot_file, dst)

    return subjects_dir, subject


def label_aal(config, src, trans=None):
    # --------------------------------------------------
    # Locate fsaverage label directory
    # --------------------------------------------------
    subject = config['source_reconstruction']['forward']['template']['subject']
    subjects_dir, subject = get_subjects_dir(subject)

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
    subject = config['source_reconstruction']['forward']['template']['subject']
    subjects_dir, subject = get_subjects_dir(subject)

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