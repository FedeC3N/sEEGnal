#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 19:21:05 2023

@author: bru
"""

import numpy
from scipy import fft


# Function for two-pass filtering.
def filtfilt(data, num=1, den=1, hilbert=False):
    ''' Filters the provided data in two passes.'''

    # Sanitizes the inputs.
    data = numpy.array(data)
    num = numpy.array(num)
    den = numpy.array(den)

    if data.ndim == 0:
        data = data.reshape(-1)
    if num.ndim == 0:
        num = num.reshape(-1)
    if den.ndim == 0:
        den = den.reshape(-1)

    # Gets the data metadata.
    dshape = data.shape
    nsample = dshape[-1]
    real = numpy.isreal(data).all()

    # Reshapes the data into a 2D array.
    data = data.reshape((-1, nsample))

    # Gets the filter metadata.
    norder = num[
        :-1,
    ].size
    dorder = den[
        :-1,
    ].size
    order = norder + dorder

    # Estimates the optimal chunk and FFT sizes.
    nfft = optnfft(nsample, order)
    chsize = nfft - 2 * order

    # Calculates the butterfly reflections of the data.
    prepad = 2 * data[:, :1] - data[:, order:0:-1]
    pospad = 2 * data[:, -1:] - data[:, -2:-order - 2:-1]

    # Adds the reflections as padding.
    paddata = numpy.concatenate((prepad, data, pospad), axis=-1)

    # Gets the Fourier transform of the filter.
    Fnum = fft.fft(num, n=nfft, axis=-1, norm=None)
    Fden = fft.fft(den, n=nfft, axis=-1, norm=None)

    # Combines all the numerators and denominators.
    if Fnum.ndim > 1:
        Fnum = Fnum.prod(axis=0, keepdims=True)
    if Fden.ndim > 1:
        Fden = Fden.prod(axis=0, keepdims=True)

    # Combines numerator and denominator.
    Ffilter = Fnum / Fden

    # Gets the squared absolute value of the filter (two-passes).
    Ffilter = Ffilter * Ffilter.conjugate()

    # Applies Hilbert transform, if required.
    if hilbert:

        # Lists the positive and negative part of the spectra.
        spos = (fft.fftfreq(nfft) > 0) & (fft.fftfreq(nfft) < 0.5)
        sneg = (fft.fftfreq(nfft) < 0) & (fft.fftfreq(nfft) > -0.5)

        # Removes the negative part of the filter spectrum.
        Ffilter[
            sneg,
        ] = 0

        # Duplicates the positive part of the filter spectrum.
        Ffilter[
            spos,
        ] = Ffilter[
            spos,
        ] * 2

        # Converts the input data into complex.
        data = data + 0j

    # Goes through each data chunk.
    for index in range(0, numpy.ceil(nsample / chsize).astype(int)):

        # Calculates the offset and length for the current chunk.
        offset = index * chsize
        chlen = numpy.min((chsize, nsample - offset))

        # Gets the chunk plus the padding.
        chunk = paddata[:, offset:offset + chlen + 2 * order].copy()

        # Takes the Fourier transform of the chunk.
        Fchunk = fft.fft(chunk, n=nfft, axis=-1, norm=None, workers=-1)

        # Applies the filter.
        Fchunk = Fchunk * Ffilter

        # Recovers the filtered chunk.
        chunk = fft.ifft(Fchunk, n=nfft, axis=-1, workers=-1)

        # Gets only the real part, if required.
        if real and not hilbert:
            chunk = chunk.real

        # Stores the filtered chunk of data.
        data[:, offset:offset + chlen] = chunk[:, order:order + chlen]

    # Restores the data shape.
    data = data.reshape(dshape)

    # Returns the filtered data.
    return data


# Function to get the optimal chunk size for the FFT.
def optnfft(nsample, order=0):
    '''Returns the optimal length of the FFT.'''

    # Uses the closest multiple of 512 * 5 samples larger than 10 filters.
    nfft = 512 * 5 * numpy.ceil((10 * order) / (512 * 5))
    nfft = nfft.astype(int)

    # If lower than 51200, uses 51200.
    nfft = numpy.max((nfft, 51200))

    # For checking with Matlab.
    nfft = numpy.min((50000, nsample)) + 2 * order

    # Returns the optimal chunck size.
    return nfft
