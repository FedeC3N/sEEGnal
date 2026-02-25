# -*- coding: utf-8 -*-
"""

This module estimates different measures as requested in config

Federico Ramírez-Toraño
24/02/2026

"""

# Imports
import sys
import traceback
from datetime import datetime as dt, timezone

import sEEGnal.feature_extraction.estimate_rel_pow

# Modules
def feature_extraction(config, BIDS):
    """

    Call one by one the needed functions for feature extraction

    :arg
    config (dict): Configuration parameters (paths, parameters, etc)
    BIDS (BIDSpath): Metadata to process

    :returns
    A dict with the result of the process

    """

    # Add the subsystem flag
    config['subsystem'] = 'feature_extraction'

    # For each measure, try to estimate the feature
    results = []
    for current_feature in config['feature_extraction']:
        try:

            # Build the function name
            func = f"estimate_{current_feature}"
            to_estimate = getattr(sEEGnal.feature_extraction.estimate_rel_pow, func)

            # Estimate the inverse solution using the defined method
            metadata = to_estimate(config, BIDS)

            # Save the results
            now = dt.now(timezone.utc)
            formatted_now = now.strftime("%d-%m-%Y %H:%M:%S")
            current_results = {
                'result': 'ok',
                'bids_basename': BIDS.basename,
                "date": formatted_now,
                'feature': current_feature,
                'metadata': metadata
            }
            results.append(current_results)

        except Exception as e:

            # Save the error
            now = dt.now(timezone.utc)
            formatted_now = now.strftime("%d-%m-%Y %H:%M:%S")
            current_results = {
                'result': 'error',
                'bids_basename': BIDS.basename,
                "date": formatted_now,
                'feature': current_feature,
                "details": f"Exception: {str(e)}, {traceback.format_exc()}"
            }
            results.append(current_results)


    return results