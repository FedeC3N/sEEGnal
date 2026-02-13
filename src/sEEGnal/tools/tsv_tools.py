# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 14:15:56 2022

@author: Ricardo
"""

import pandas


def write_tsv(table, filename):

    # If the input is a dictionary, converts it into a Pandas data frame.
    if isinstance(table, dict):
        table = pandas.DataFrame(table['matrix'], index=table['rows'], columns=table['columns'])

        # Marks the data as a matrix.
        ismatrix = True

    else:
        ismatrix = False

    # Checks that the input is a Pandas data frame.
    assert isinstance(table, pandas.DataFrame), 'Invalid input data.'

    # Writes the data as a TSV file.
    table.to_csv(filename, sep='\t', na_rep='n/a', index=ismatrix, index_label='output_name', encoding='utf-8-sig')


# Helper function to read TSV data.
def read_tsv(filename, ismatrix=False):

    # If the TSV contains a matrix the first column is the row label.
    if ismatrix:
        index_col = 0
    else:
        index_col = None

    # Reads the table as a DataFrame object.
    table = pandas.read_table(filename, delimiter='\t', index_col=index_col, comment=None, encoding='utf-8-sig')

    # If the TSV contains a matrix rewrites the output.
    if ismatrix:
        table = {'rows': table.index, 'columns': table.columns, 'matrix': table.values}

    # Returns the read table.
    return table