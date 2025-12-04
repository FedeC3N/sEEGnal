# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 00:32:54 2022

@author: Ricardo Bruña
"""

import numpy


# Function to read RIFF and RF64 files.
def read_file(filename):

    # Opens the file to read.
    with open(filename, 'rb') as fid:

        # Gets the total file length.
        fid.seek(0, 2)
        flen = fid.tell()
        fid.seek(0, 0)

        # Reads the magic number.
        mnum = numpy.fromfile(fid, (bytes, 4), 1)[0].decode()

        # Defines the length of the pointer (chunk size marker).
        if mnum == 'RIFF':
            plen = 4
        elif mnum == 'RF64':
            plen = 8
        else:
            raise ValueError('Not a valid RIFF file.')

        # Reads the total data length.
        dlen = numpy.fromfile(fid, f'<u{plen}', 1)[0]

        # If the data extends beyond the file rises an error.
        if dlen + plen + 4 > flen:
            raise EOFError('Data is incomplete or extends beyond the end of the file.')

        # Restarts the file cursor.
        fid.seek(0, 0)

        # Reads the RIFF tree from the file.
        tree = read_tree(fid, plen)

        # Returns the tree.
        return tree


# Function to read a RIFF tree.
def read_tree(fid, plen):

    # Reads the label and length for the current chunk.
    clab = numpy.fromfile(fid, (bytes, 4), 1)[0].decode()
    clen = numpy.fromfile(fid, f'<u{plen}', 1)[0]
    dpos = fid.tell()

    # If the chunk is a list entry iterates over it.
    if set.issubset({clab}, {'RIFF', 'RF64', 'LIST'}):

        # Reads the block label.
        clab = numpy.fromfile(fid, (bytes, 4), 1)[0].decode()
        clen = clen - 4

        # Gets the current position of the pointer.
        cpos = fid.tell()

        # Initilizes the data field.
        data = []
        chil = []

        # Iterates through the subtrees.
        while True:

            # Checks if the chunk is completely read.
            if fid.tell() >= cpos + clen:
                break

            # Reads the next child.
            chil = chil + [read_tree(fid, plen)]

    elif len(fid.peek(1)) > 0:

        # Reads the data.
        data = numpy.fromfile(fid, 'uint8', clen)

        # Reads the padding data, if required.
        numpy.fromfile(fid, 'uint8', int(clen % 2))

        # Initializes the children structure.
        chil = []

    # Creates the tree structure.
    tree = {'label': clab, 'length': clen, 'datapos': dpos, 'data': data, 'children': chil}

    # Returns the tree.
    return tree


# Function to get a branch of the RIFF tree.
def get_subtree(tree, blabs=None):

    if (blabs is None):
        blabs = []

    # Crecks the input.
    assert isinstance(blabs, list), 'Second input must be a list of branches.'

    # If no branch defined returns the current one.
    if len(blabs) == 0:
        return tree

    # Lists the branches in the tree.
    brans = [bran['label'].strip() for bran in tree['children']]

    # Gets the label for the first branch.
    blab = blabs[0]

    # Looks for the branch.
    index = numpy.flatnonzero(numpy.in1d(brans, blab))

    assert (len(index) > 0), 'Branch not found.'

    if len(index) > 1:
        raise UserWarning('Several hits for the branch, returning only the first result.')

    # Keeps only the desired branch.
    tree = tree['children'][index[0]]

    # Iterates through the branch, if required.
    tree = get_subtree(tree, blabs[1:])

    # Returns the requested subtree.
    return tree
