# -*- coding: utf-8 -*-
"""
Functional Connectivity utilities for sEEGnal

Federico Ramírez-Toraño
26/02/2026
"""

# Imports
import numpy


def compute_plv(data=None,dtype=numpy.complex64):
    """
    Based on Ricardo Bruña: Phase locking value revisited: teaching new tricks to an old dog
    Underlying logic:
    The more time-consuming calculation is the complex exponential.
    This algorithm works in the exponential space the whole time.
    Uses the logarithmic operations for the phase difference:
    exp ( 1i * ( a - b ) ) = exp ( 1i * a ) / exp ( 1i * b )
    Performs all the divisions at once with matrix multiplication.
    """

    # Allocate memory for output
    nepochs, nsources, nsamples = data.shape

    # Reserve memory
    plv = numpy.zeros((nsources, nsources, nepochs), dtype=dtype)

    # Loop over trials
    for iepoch in range(nepochs):

        sourcedata = data[iepoch,:,:]  # (nsources, nsamples)

        # Avoid division by zero
        abs_data = numpy.abs(sourcedata)
        abs_data[abs_data == 0] = 1e-12

        # Extract phase vector (e^{iφ})
        sourcevector = sourcedata / abs_data

        # Complex PLV
        plv[:, :, iepoch] = (
                                   sourcevector @ numpy.conjugate(sourcevector.T)
                           ) / nsamples

    # Remove diagonal (NaN)
    diag_idx = numpy.arange(nsources)
    plv[diag_idx, diag_idx, :] = numpy.nan

    # Get the PLV as the module of the complex vector
    plv = numpy.abs(plv)

    # Vectorize upper-triangular (excluding diagonal)
    triu_idx = numpy.triu_indices(nsources, k=1)
    plv_vector = plv[triu_idx[0], triu_idx[1], :].T  # shape (n_epochs, n_connections)

    return plv_vector


def reconstruct_plv_matrix(plv_vector, conn_indices, n_sources):
    mat = numpy.zeros((n_sources, n_sources), dtype=plv_vector.dtype)
    mat[conn_indices] = plv_vector
    mat[(conn_indices[1], conn_indices[0])] = plv_vector
    return mat
