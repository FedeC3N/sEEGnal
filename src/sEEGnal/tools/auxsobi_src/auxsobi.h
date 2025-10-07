/* Based on function:
 * * sobi.m by A. Belouchrani and A. Cichocki.
 *
 * Based on publications:
 * * Belouchrani, Abed-Meraim, Cardoso & Moulines 1993
 *   Proc. Int. Conf. on Digital Sig. Proc. 346-351.
 * * Belouchrani, Abed-Meraim 1993
 *   Proc. Gretsi, (Juan-les-pins) 309-312.
 * * Cichocki & Amari 2003
 *   Adaptive Blind Signal and Image Processing, Wiley.
 * * JJ Polcari
 *   Closed Form SVD Solutions for 2 x 2 Matrices - Rev 2
 *   https://www.researchgate.net/publication/263580188
 */


/* Shortcut for the sign of a value. */
double
sign ( const double value ) {
    return ( value < 0 ) ? -1 : +1;
}



/* Function to calculate the dot product between two vectors using a long double accumulator. */
double
long_dot ( double * data1, double * data2, long int length ) {

    long double tmp = 0;
    double result;
    long int index;


    /* Accumulates the product of the elements. */
    for ( index = 0; index < length; index ++ )
        tmp += data1 [ index ] * data2 [ index ];

    /* Re-casts the result into double precision. */
    result = (double) tmp;

    /* Returns the result of the dot product. */
    return result;
}



/* Function to calculate the dot product between two vectors. */
double
normal_dot ( double * data1, double * data2, long int length ) {

    double result = 0;
    long int index;


    /* Accumulates the product of the elements. */
    for ( index = 0; index < length; index ++ )
        result += data1 [ index ] * data2 [ index ];

    /* Returns the result of the dot product. */
    return result;
}


/* Function to calculate the SVD of a masked 3x3 matrix. */
void
masked_svd (
    double * c, double * sr, double * sc,
    double g1g1, double g2g2, double g3g3, double g1g2 ) {

    /*
    Function to calculate the rotation between components for SOBI.

    Based on calculating the main eigenvector of the masked matrix:
    | g1g1 g1g2    0 |
    | g2g1 g2g2    0 |
    |    0    0 g3g3 |

    Based on paper:
    * "Closed Form SVD Solutions for 2 x 2 Matrices - Rev 2" by JJ Polcari.
      https://www.researchgate.net/publication/263580188
    */

    double tmp1, tmp2;
    double eigval, cosTh, sinTh;


    /* Calculates the main eigenvalue of the sub-matrix (Eq. 3). */
    tmp1    = sqrt ( ( g1g1 - g2g2 ) * ( g1g1 - g2g2 ) + 4 * g1g2 * g1g2 );
    eigval  = 0.5 * ( ( g1g1 + g2g2 ) + tmp1 );

    /* If g3g3 is larger, there is no rotation to apply. */
    if ( eigval < g3g3 ) {
        *c       = 1;
        *sc = *sr = 0;

    /* Otherwise uses the 2x2 matrix. */
    } else {

        /* Calculates the main eigenvector of the 2x2 matrix (Eq. 6). */
        tmp2    = ( g1g1 - g2g2 ) / tmp1;
        cosTh   = sqrt ( 0.5 * ( 1 + tmp2 ) );
        //sinTh   = sign ( g1g2 ) * sqrt ( 0.5 * ( 1 - tmp2 ) );

        /* Calculates the terms for the Givens rotation matrix. */
        *c       = sqrt ( 0.5 * ( 1 + cosTh ) );
        //*sr = *sc = 0.5 * ( sinTh ) / c;

        /* Makes sure that the square sum of c and sr/sc is 1. */
        *sr = *sc = sign ( g1g2 ) * sqrt ( 1 - *c * *c );
    }
}



/* Function to apply a Givens rotation to a 3D matrix (nsrc x nsrc x nlag). */
void
apply_rot_3d (
    double * data, long int nsrc, long int nlag,
    long int p, long int q,
    double c, double sr, double sc ) {

    long int offp, offq, ind1, ind2;
    double tmpp, tmpq;


    /* Iterates through the columns. */
    for ( ind1 = 0; ind1 < nsrc; ind1 ++ ) {

        /* Calculates the offsets. */
        offp = ind1 * nsrc * nlag + p * nlag;
        offq = ind1 * nsrc * nlag + q * nlag;

        /* Iterates through the elements. */
        for ( ind2 = 0; ind2 < nlag; ind2 ++ ) {

            /* Gets the initial data. */
            tmpp = data [ offp + ind2 ];
            tmpq = data [ offq + ind2 ];

            /* Applies the rotation. */
            data [ offp + ind2 ] = c * tmpp + sr * tmpq;
            data [ offq + ind2 ] = c * tmpq - sc * tmpp;
        }
    }

    /* Iterates through the rows. */
    for ( ind1 = 0; ind1 < nsrc; ind1 ++ ) {

        /* Calculates the offsets. */
        offp = p * nsrc * nlag + ind1 * nlag;
        offq = q * nsrc * nlag + ind1 * nlag;

        /* Iterates through the elements. */
        for ( ind2 = 0; ind2 < nlag; ind2 ++ ) {

            /* Gets the initial data. */
            tmpp = data [ offp + ind2 ];
            tmpq = data [ offq + ind2 ];

            /* Applies the rotation. */
            data [ offp + ind2 ] = c * tmpp + sc * tmpq;
            data [ offq + ind2 ] = c * tmpq - sr * tmpp;
        }
    }
}



/* Function to apply a Givens rotation to a 2D matrix (nsrc x nlag). */
void
apply_rot_2d (
    double * data, long int nlag,
    long int p, long int q,
    double c, double sr, double sc ) {

    long int offp, offq, ind;
    double tmpp, tmpq;


    /* Calculates the offsets. */
    offp = p * nlag;
    offq = q * nlag;

    /* Iterates through the elements. */
    for ( ind = 0; ind < nlag; ind ++ ) {

        /* Gets the initial data. */
        tmpp = data [ offp + ind ];
        tmpq = data [ offq + ind ];

        /* Applies the rotation. */
        data [ offp + ind ] = c * tmpp + sr * tmpq;
        data [ offq + ind ] = c * tmpq - sc * tmpp;
    }
}
