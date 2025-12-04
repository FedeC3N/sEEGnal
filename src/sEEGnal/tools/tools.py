# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:24:00 2024

@author: Ricardo
"""

import numpy


# Helper function to find matches of one array in another.
def find_matches(array, target):

    # Converts the inputs into Numpy arrays, if required.
    array = numpy.asarray(array)
    target = numpy.asarray(target)

    # Initializes the array of matches.
    matches = [None] * len(array)

    # Iterates through the array.
    for index in range(array.size):

        # Looks for the match.
        match = numpy.flatnonzero(target == array[index])

        # Stores the match, if any.
        if match.size > 0:
            matches[index] = match[0]

    # Returns the matches.
    return matches
