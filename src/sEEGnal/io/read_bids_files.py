"""
Read BIDS EEG and events

Created on 15/04/2026

Federico Ramirez-Toraño
"""

import mne_bids

def read_BIDS_files(BIDS, preload=True):

    raw, event_id = mne_bids.read_raw_bids(
        BIDS,
        return_event_dict=True,
        verbose=False
    )

    if preload:
        raw.load_data()

    return raw, event_id