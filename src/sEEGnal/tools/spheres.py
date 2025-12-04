# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:24:00 2024

@author: Ricardo
"""

import numpy
import scipy

import mne


def spline_int(elec1, elec2=None, order=4, degree=None, lambda_=1e-5):
    """
    Based on FieldTrip 20200130 functions:
    * splint by Robert Oostenveld

    Based on papers:
    * Perrin et al. 1989 Electroencephalogr. Clin. Neurophysiol. 72.184-7.
      doi: 10.1016/0013-4694(89)90180-6
    * Perrin et al. 1990 Electroencephalogr. Clin. Neurophysiol. 76.565.
      doi: 10.1016/0013-4694(90)90009-9
    """

    # If no second set of channels, copies the first one.
    if elec2 is None:
        elec2 = elec1.copy()

    # Checks the input.
    elec1 = check_elec(elec1)
    elec2 = check_elec(elec2)

    # Gets the size of the channel sets.
    siz1 = len(elec1)
    siz2 = len(elec2)

    # Sets the default degree according to the number of real channels.
    if degree is None:
        if siz1 <= 32: degree = 9
        elif siz1 <= 64: degree = 14
        elif siz1 <= 128: degree = 20
        else: degree = 32

    # Gets the raw EEG positions.
    pos1 = numpy.vstack([pos for pos in elec1.values()])
    pos2 = numpy.vstack([pos for pos in elec2.values()])

    # Concatenates all the electrodes (real and reconstructed).
    poss = numpy.vstack((pos1, pos2))

    # Gets only one instance per electrode.
    poss, inds = numpy.unique(poss, return_inverse=True, axis=0)
    ind1 = inds[:siz1]
    ind2 = inds[siz1:]

    # Identifies the center of the electrodes and centers them.
    cent, radi = fit_sphere(poss)
    poss = poss - cent

    # Projects the electrodes into a unit sphere.
    poss = poss / numpy.sqrt(numpy.sum(poss**2, 1, keepdims=True))

    # Gets the angle between electrodes.
    CosE = numpy.dot(poss, poss[ind1].T)
    CosE = numpy.minimum(numpy.maximum(CosE, -1), 1)

    # Solves the g(x) and h(x) equations (Eqs. 3 and 5b).
    gx, hx = perrin_gh(order, degree, CosE)

    # Gets the g(x) submatrix for the input set.
    gxii = gx[ind1, :]

    # Builds a composite matrix (with Tikhonov regularization of g(x)).
    H = numpy.ones((siz1 + 1, siz1 + 1))
    H[0, 0] = 0
    H[1:, 1:] = gxii + numpy.eye(siz1) * lambda_

    # Inverts the matrix to fit the solution for the input.
    iH = numpy.linalg.pinv(H)

    # Gets the g(x) and h(x) submatrices for the output set.
    gxio = gx[ind2, :]
    hxio = hx[ind2, :]

    # Computes the filter to interpolate the potential in the output set.
    dummy = numpy.hstack((numpy.ones((siz2, 1)), gxio))
    WVo = numpy.dot(dummy, iH[:, 1:])

    # Computes the filter to interpolate the Laplacian in the output set.
    WLo = numpy.dot(hxio, iH[1:, 1:])
    WLo = WLo / (radi**2)

    return WVo


def fit_sphere(poss):
    '''
    Based on FieldTrip 20160222 (from SPM) functions:
    * fitsphere by Jean Daunizeau
    '''

    # Gets the number of points.
    npos = poss.shape[0]

    # Adds the absolute distance to each point.
    dist = numpy.sum(poss**2, axis=1)
    dum = numpy.hstack((dist[:, None], poss))

    # Creates the design matrices.
    M = numpy.zeros((5, 5))
    M[:4, :4] = numpy.dot(dum.T, dum)
    M[4, :4] = numpy.sum(dum, axis=0)
    M[:4, 4] = numpy.sum(dum, axis=0)
    M[4, 4] = npos

    N = numpy.eye(5) * npos
    N[0, 0] = 4 * numpy.sum(dist)
    N[1:4, 0] = 2 * numpy.sum(poss, axis=0)
    N[0, 1:4] = 2 * numpy.sum(poss, axis=0)
    N[4, 4] = 0

    # Gets the eigenvalues and eigenvectors of M.
    val, vec = numpy.linalg.eig(M)

    # If the matrix is full-rank uses the largest eigenvector.
    tiny = numpy.finfo(float).eps * 5 * numpy.linalg.norm(M)
    if numpy.all(val > tiny):

        dum = numpy.dot(numpy.linalg.inv(M), N)
        val, vec = numpy.linalg.eig(dum)
        pvec = vec[:, numpy.argmax(val)]

    # If it is rank-deficient extracts the null space.
    else:

        pvec = vec[:, numpy.argmin(val)]

    # Calculates the center and the radius of the best-fitting sphere.
    cent = -pvec[1:4] / pvec[0] / 2
    radi = numpy.sqrt(numpy.sum(cent**2) - pvec[4] / pvec[0])

    # Returns the center and the radius.
    return cent, radi


def perrin_gh(order, degree, x):
    """
    Based on FieldTrip 20200130 functions:
    * splint/gh by Robert Oostenveld
    
    Based on papers:
    * Perrin et al. 1989 Electroencephalogr. Clin. Neurophysiol. 72.184-7.
      doi: 10.1016/0013-4694(89)90180-6
    * Perrin et al. 1990 Electroencephalogr. Clin. Neurophysiol. 76.565.
      doi: 10.1016/0013-4694(90)90009-9
    """

    # Calculates the Legendre polynomials up to the requested degree.
    pol = plgndr(degree, 0, x)
    pol = pol[:, 1:]

    # Gets the values of g(x) and h(x) (Eqs. 3 and 5b).
    vdeg = numpy.arange(1, degree + 1)
    gx = numpy.sum((2 * vdeg + 1) / (vdeg * (vdeg + 1))**order * pol, 1) / (4 * numpy.pi)
    hx = -numpy.sum((2 * vdeg + 1) / (vdeg * (vdeg + 1))**(order - 1) * pol, 1) / (4 * numpy.pi)

    # Reshapes g(x) and h(x) as matrices.
    gx.shape = x.shape
    hx.shape = x.shape

    return gx, hx


def plgndr(degree, order, x):
    """
    Based on FieldTrip 20160222 functions:
    * plgndr
    """

    # Checks the input.
    assert order == numpy.round(order) and order >= 0, 'The order must be a positive integer or zero.'
    assert degree == numpy.round(
        degree
    ) and degree >= order, 'The degree must be a positive integer equal or larger than the order.'
    assert numpy.all(numpy.abs(x) <= 1), 'Absolute value of x cannot be larger than 1.'

    # Initializes the output.
    pol = numpy.zeros((x.size, degree + 1))

    # Flattens the input.
    x = x.flatten()

    # Gets the solution for degree=order analytically.
    root = numpy.sqrt(1 - x * x)
    pol[:, order] = numpy.prod(numpy.arange(1, 2, 2 * order - 1)) * (root**order)

    # Fixes the sign, if required.
    if abs(order) % 2 != 0:
        pol[:, order] = -pol[:, order]

    # If the degree is larger than the order, gets the rest of the terms.
    if degree > order:

        # Uses a simplified formula for the second terms.
        pol[:, order + 1] = x * (2 * (order + 1) - 1) * pol[:, order]

        # Uses the iterative formula for the rest of terms.
        for index in range(order + 2, degree + 1):

            pol1 = x * (2 * index - 1) * pol[:, index - 1]
            pol2 = (index + order - 1) * pol[:, index - 2]
            pol[:, index] = (pol1 - pol2) / (index - order)

    return pol


def check_elec(elec):

    # Checks if the input is a montage object.
    if isinstance(elec, mne.channels.DigMontage):
        elec = elec.get_positions()['ch_pos']

    # Checks that the electrode definition is a dictionary.
    assert isinstance(elec, dict), 'Input must be a channel position dictionary.'

    # Checks whether all entries in the dictionary are valid (3,) float arrays.
    assert all(entry.shape == (3, ) for entry in elec.values()), 'The electrode postiions must be (3,) arrays.'
    assert all(entry.dtype == float for entry in elec.values()), 'The electrode postiions must be (3,) arrays.'

    return elec
