#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 18:53:36 2023

@author: bru
"""

import math
import time
import scipy, scipy.linalg
import numpy

import sEEGnal.tools.auxsobi as auxsobi


def sobi(data, nlag=None, nsource=None):
    '''
    Based on function:
        * sobi.m by A. Belouchrani and A. Cichocki.
    
    Based on publications:
        * Belouchrani, Abed-Meraim, Cardoso & Moulines 1993
          Proc. Int. Conf. on Digital Sig. Proc. 346-351.
        * Belouchrani, Abed-Meraim 1993
          Proc. Gretsi, (Juan-les-pins) 309-312.
        * Cichocki & Amari 2003
          Adaptive Blind Signal and Image Processing, Wiley.
    '''

    # Gets the metadata (assumes data in MNE-Python shape).
    nchannel = data.shape[-2]
    nsample = data.shape[-1]
    data = data.reshape((-1, nchannel, nsample))
    nepoch = data.shape[0]

    # If no requested otherwise, sets the number of lags to 1/3 of the data.
    if nlag is None:
        nlag = min(100, numpy.ceil(nsample / 3))

    # If no requested otherwise, uses as many sources as channels.
    if nsource is None:
        nsource = nchannel

    # Rewrites the data as channels by samples by epochs.
    data = numpy.moveaxis(data, 0, -1).copy()

    # Demeans the data.
    data = data - data.mean(axis=1, keepdims=True)

    # Concatenates all the epochs.
    data = data.reshape((data.shape[0], -1))

    # Calculates a whitener using SVD.
    [u, s, v] = scipy.linalg.svd(numpy.dot(data, data.T))

    # Restricts the number of sources, if requested.
    u = u[:, :nsource]
    s = s[:nsource]

    # Calculates the whitener and unwhitener,
    wfilter = (u / (numpy.sqrt(s))).T
    uwfilter = scipy.linalg.pinv(wfilter)

    # Whitens the data.
    wdata = numpy.dot(wfilter, data)

    # Restores the original shape of the data.
    wdata = wdata.reshape((nchannel, nsample, nepoch))

    # Rewrites the whitened data as samples by epochs by sources.
    wdata = numpy.moveaxis(wdata, 0, -1).copy()

    # Initializes the correlations matrix.
    xcorr = numpy.zeros((nchannel, nchannel, nlag))

    # Gets the correlation matrix for each lag.
    for lag in range(nlag):

        # Gets the overlapping data for the current lag.
        wdata1 = wdata[lag + 1:, :, :].reshape((-1, nsource))
        wdata2 = wdata[:nsample - lag - 1, :, :].reshape((-1, nsource))

        # Gets the unbiased normalized correlation.
        Rxp = numpy.dot(wdata1.T, wdata2)
        Rxp = Rxp / nepoch / (nsample - lag - 1)
        Rxp = Rxp * scipy.linalg.norm(Rxp)

        # Stores the normalized correlation for this lag.
        xcorr[:, :, lag] = Rxp

    # Initializes the flags and counters.
    iter_ = 0
    loop = True

    # Defines the stop condition for the iterations.
    eps = 1 / numpy.sqrt(nsample) / 100
    eps2 = eps**2

    # Initializes the mixing matrix to the identity.
    wmixing = numpy.eye(nchannel)

    # Performs the joint diagonalization.
    start_SOBI = time.time()
    while loop:

        # Updates the iteration index and reset the flags.
        iter_ += 1
        loop = False

        #print ( iter_ )

        # Check the elapsed time
        current_time_SOBI = time.time()
        elapsed_time_current_iteration = (current_time_SOBI - start_SOBI)

        # Raise an error if the SOBI process takes more than 6 hours
        if elapsed_time_current_iteration > 6 * 60 * 60:
            raise Exception('SOBI process aborted. It took more than 6 hours')

        # Goes through each pair of sources.
        for p in range(nchannel - 1):
            for q in range(p + 1, nchannel):

                # Computes the dependence between the sources.
                g1 = xcorr[p, p, :] - xcorr[q, q, :]
                g2 = xcorr[p, q, :] + xcorr[q, p, :]
                g3 = xcorr[q, p, :] - xcorr[p, q, :]

                # Gets the Givens rotation to maximize independence between sources.
                #c, sr, sc = sobi_get_rot ( g1, g2, g3 )
                #c, sr, sc = sobi_get_fastrot ( g1, g2, g3 )
                c, sr, sc = auxsobi.get_rot(g1, g2, g3)

                # Performs the rotation, if required.
                if sc * sr > eps2:

                    # At least one element was modified, so iterates again.
                    loop = True

                    # Rotates the cross-correlation matrix.
                    #xcorr    = sobi_apply_rot_3d ( xcorr, p, q, c, sr, sc )
                    auxsobi.apply_rot(xcorr, p, q, c, sr, sc)

                    # Rotates the mixing matrix.
                    #wmixing  = sobi_apply_rot_2d ( wmixing, p, q, c, sr, sc )
                    auxsobi.apply_rot(wmixing, p, q, c, sr, sc)

    # Un-whitens the mixing matrix.
    mixing = numpy.dot(uwfilter, wmixing.T)

    # Estimates the unmixing matrix.
    unmixing = scipy.linalg.pinv(mixing)

    # Returns the mixing and unmixing matrices.
    return mixing, unmixing


# Defines the matrix mask for the rotation.
mask = numpy.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])


# Function to get the Givens rotation for SOBI.
def sobi_get_rot(g1, g2, g3):
    """
    Get the main eigenvector of the masked matrix:
    | g1g1 g1g2    0 |
    | g2g1 g2g2    0 |
    |    0    0 g3g3 |

    :param g1:
    :param g2:
    :param g3:
    :return:
    """

    # Gets the masked matrix.
    g = numpy.array([g1, g2, g3])
    gg = numpy.dot(g, g.T)
    gg = gg * mask

    # Gets the main eigenvector.
    angles = scipy.linalg.svd(gg)[0]
    angles = angles[:, 0]

    # Checks if the main eigenvector includes the first columns/rows.
    if abs(angles[0]) > 0:

        # Makes sure that the eigenvector has the right sign.
        angles = math.copysign(1, angles[0]) * angles

        # Gets the Givens rotation to maximize independece.
        c = math.sqrt(0.5 * (1 + angles[0]))
        #sr      = 0.5 * ( angles [1] - 1j * angles [2] ) / c
        #sc      = 0.5 * ( angles [1] + 1j * angles [2] ) / c
        sr = sc = 0.5 * (angles[1]) / c

    # Otherwise, no rotation is applied.
    else:
        c = 1
        sr = sc = 0

    # Returns the rotation matrix.
    return c, sr, sc


# Function to get the Givens rotation for SOBI.
def sobi_get_fastrot(g1, g2, g3):
    """
    Get the main eigenvector of the masked matrix:
    | g1g1 g1g2    0 |
    | g2g1 g2g2    0 |
    |    0    0 g3g3 |

    :param g1:
    :param g2:
    :param g3:
    :return:

    Based on paper:
    * "Closed Form SVD Solutions for 2 x 2 Matrices - Rev 2" by JJ Polcari.
      https://www.researchgate.net/publication/263580188
    """

    # Calculates the projection of the inputs.
    g1g1 = g1.dot(g1)
    g2g2 = g2.dot(g2)
    g1g2 = g1.dot(g2)
    g3g3 = g3.dot(g3)
    '''
    We need the SVD of the masked matrix:
    | x1x1 x1x2    0 |
    | x2x1 x2x2    0 |
    |    0    0 x3x3 |

    The largest eigenvalue is the largest of:
    * the largest eigenvalue of the G1-2 submatrix,
    * x3x3.
    '''

    # Gets the largest eigenvalue of G1-2 using the analytical SVD (Eq. 3).
    sqrt = math.sqrt((g1g1 - g2g2) * (g1g1 - g2g2) + 4 * g1g2 * g1g2)
    maxeig = 0.5 * ((g1g1 + g2g2) + sqrt)

    # If the largest eigenvalue is that of G1-2.
    if maxeig > g3g3:

        # Gets the main eigenvector using the analytical SVD (Eq. 6).
        sb = math.copysign(1, g1g2)
        dummy = (g1g1 - g2g2) / sqrt
        cosTh = math.sqrt(0.5 * (1 + dummy))
        sinTh = sb * math.sqrt(0.5 * (1 - dummy))

        # Gets the Givens rotation matrix.
        c = math.sqrt(0.5 * (1 + cosTh))
        sr = sc = 0.5 * (sinTh) / c

        # Makes sure that the square sum of c and sr/sc is 1.
        #sr = sc = sb * math.sqrt ( 1 - c ** 2 )

    # If the largest eigenvalue is x3x3.
    else:
        c = 1
        sr = sc = 0

    # Returns the rotation matrix.
    return c, sr, sc


def sobi_apply_rot_3d(matrix, p, q, c, sr, sc):

    # Rotates the matrix matrix along the columns.
    dummyp = matrix[:, p, :].copy()
    dummyq = matrix[:, q, :].copy()
    matrix[:, p, :] = c * dummyp + sr * dummyq
    matrix[:, q, :] = c * dummyq - sc * dummyp

    # Rotates the matrix matrix along the rows.
    dummyp = matrix[p, :, :].copy()
    dummyq = matrix[q, :, :].copy()
    matrix[p, :, :] = c * dummyp + sc * dummyq
    matrix[q, :, :] = c * dummyq - sr * dummyp

    # Returns the rotated matrix.
    return matrix


def sobi_apply_rot_2d(matrix, p, q, c, sr, sc):

    # Rotates the matrix.
    dummyp = matrix[p, :].copy()
    dummyq = matrix[q, :].copy()
    matrix[p, :] = c * dummyp + sr * dummyq
    matrix[q, :] = c * dummyq - sc * dummyp

    # Returns the rotated matrix.
    return matrix
