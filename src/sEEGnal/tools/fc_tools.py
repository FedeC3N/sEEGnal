# -*- coding: utf-8 -*-
"""
Functional Connectivity utilities for sEEGnal

Federico Ramírez-Toraño
26/02/2026
"""

# Imports
import numpy
from scipy.signal import hilbert


def compute_plv(data=None,average_epochs=True,dtype=numpy.complex64):
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
        sourcedata = data[iepoch, :, :]  # (nsources, nsamples)

        # Compute analytic signal to extract phase
        analytic_signal = hilbert(sourcedata, axis=-1)  # señal compleja
        sourcevector = analytic_signal / numpy.abs(analytic_signal)  # e^{iφ}, fase

        # Complex PLV
        plv[:, :, iepoch] = (
                                    sourcevector @ numpy.conj(sourcevector.T)
                            ) / sourcedata.shape[1]

    # Remove diagonal (NaN)
    diag_idx = numpy.arange(nsources)
    plv[diag_idx, diag_idx, :] = numpy.nan

    # Get the PLV as the module of the complex vector
    plv = numpy.abs(plv)

    # Vectorize upper-triangular (excluding diagonal)
    triu_idx = numpy.triu_indices(nsources, k=1)
    plv_vector = plv[triu_idx[0], triu_idx[1], :].T  # shape (n_epochs, n_connections)

    if average_epochs:
        plv_vector = numpy.mean(plv_vector, axis=0)

    return plv_vector


def compute_ciplv(data=None, average_epochs=True, dtype=numpy.complex64):
    """
    Based on Ricardo Bruña: Phase locking value revisited.
    Computes ciPLV inside the epoch loop.
    """

    # Allocate memory for output (now float, not complex)
    nepochs, nsources, nsamples = data.shape

    # Reserve memory for ciPLV directly
    ciplv = numpy.zeros((nsources, nsources, nepochs), dtype=numpy.float32)

    # Loop over trials
    for iepoch in range(nepochs):
        sourcedata = data[iepoch, :, :]  # (nsources, nsamples)

        # Compute analytic signal to extract phase
        analytic_signal = hilbert(sourcedata, axis=-1)  # señal compleja
        sourcevector = analytic_signal / numpy.abs(analytic_signal)  # e^{iφ}, fase

        # Complex PLV (temporary)
        cplv = (sourcevector @ numpy.conj(sourcevector.T)) / sourcedata.shape[1]

        # Remove diagonal (set to zero before ciPLV computation)
        numpy.fill_diagonal(cplv, 0)

        # Estimate the real and imaginary part
        real_plv = numpy.real(cplv)
        imag_plv = numpy.imag(cplv)
        denom = numpy.sqrt(1 - real_plv ** 2)
        denom[denom == 0] = numpy.finfo(numpy.float32).eps

        # Estimate ciPLV
        ciplv_epoch = numpy.abs(imag_plv / denom)

        # Store result for this epoch
        ciplv[:, :, iepoch] = ciplv_epoch.astype(numpy.float32)

    # Set diagonal to NaN (like your original code)
    diag_idx = numpy.arange(nsources)
    ciplv[diag_idx, diag_idx, :] = numpy.nan

    # Vectorize upper-triangular (excluding diagonal)
    triu_idx = numpy.triu_indices(nsources, k=1)
    ciplv_vector = ciplv[triu_idx[0], triu_idx[1], :].T  # (n_epochs, n_connections)

    if average_epochs:
        ciplv_vector = numpy.mean(ciplv_vector, axis=0)

    return ciplv_vector


def reconstruct_fc_matrix(plv_vector, conn_indices, n_sources):
    mat = numpy.zeros((n_sources, n_sources), dtype=plv_vector.dtype)
    mat[conn_indices] = plv_vector
    mat[(conn_indices[1], conn_indices[0])] = plv_vector
    return mat
