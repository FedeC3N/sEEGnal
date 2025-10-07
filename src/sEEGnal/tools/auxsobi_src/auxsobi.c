#define PY_SSIZE_T_CLEAN
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "auxsobi.h"



/* Function to calculate the Givens rotation to maximize the independence of two sources. */
static PyObject *
Pysobi_rot ( PyObject *self, PyObject *args ) {

    double *d1, *d2, *d3;
    double g1g1, g2g2, g3g3, g1g2;
    double c, sr, sc;

    PyObject *i1, *np1, *i2, *np2, *i3, *np3;


    /* Gets the inputs. */
    if ( !PyArg_ParseTuple ( args, "OOO", &i1, &i2, &i3 ) )
        return NULL;

    /* Extracts the NumPy arrays from the input. */
    np1 = PyArray_FROM_OTF ( i1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY );
    np2 = PyArray_FROM_OTF ( i2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY );
    np3 = PyArray_FROM_OTF ( i3, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY );


    /* Checks the data type of the inputs. */
    if ( np1 == NULL ||np2 == NULL || np3 == NULL ) {
        PyErr_SetString ( PyExc_TypeError, "Inputs must be one-dimensional NumPy arrays." );
        PyErr_Occurred ();
        return NULL;
    }

    /* Checks that the entries are 1D arrays. */
    if ( PyArray_NDIM ( np1 ) != 1 || PyArray_NDIM ( np2 ) != 1 || PyArray_NDIM ( np3 ) != 1 ) {
        PyErr_SetString ( PyExc_TypeError, "Inputs must be one-dimensional NumPy arrays." );
        PyErr_Occurred ();
        return NULL;
    }

    /* Checks that the three arrays have the same length. */
    if ( PyArray_DIMS ( np1 ) [0] != PyArray_DIMS ( np2 ) [0] || PyArray_DIMS ( np1 ) [0] != PyArray_DIMS ( np3 ) [0] ) {
        PyErr_SetString ( PyExc_TypeError, "Input arrays have different lengths." );
        PyErr_Occurred ();
        return NULL;
    }


    /* Gets the data. */
    d1 = (double *) PyArray_DATA ( np1 );
    d2 = (double *) PyArray_DATA ( np2 );
    d3 = (double *) PyArray_DATA ( np3 );

    /* Calculates the entries of the matrix. */
    g1g1 = long_dot ( d1, d1, PyArray_DIMS ( np1 ) [0] );
    g2g2 = long_dot ( d2, d2, PyArray_DIMS ( np1 ) [0] );
    g3g3 = long_dot ( d3, d3, PyArray_DIMS ( np1 ) [0] );
    g1g2 = long_dot ( d1, d2, PyArray_DIMS ( np1 ) [0] );

    /* Frees the NumPy array objects. */
    Py_DECREF ( np1 );
    Py_DECREF ( np2 );
    Py_DECREF ( np3 );


    /* Calculates the Givens rotation using SVD. */
    masked_svd ( &c, &sr, &sc, g1g1, g2g2, g3g3, g1g2 );


    /* Returns the rotation matrix in a tuple. */
    return Py_BuildValue ( "ddd", c, sr, sc );
}



/* Function to calculate the SVD of a masked 3x3 matrix. */
static PyObject *
Pysobi_svd ( PyObject *self, PyObject *args ) {

    double g1g1, g2g2, g3g3, g1g2;
    double c, sr, sc;


    /* Gets the inputs. */
    if ( !PyArg_ParseTuple ( args, "dddd", &g1g1, &g2g2, &g3g3, &g1g2 ) )
        return NULL;


    /* Calculates the Givens rotation using SVD. */
    masked_svd ( &c, &sr, &sc, g1g1, g2g2, g3g3, g1g2 );

    /* Returns the rotation matrix in a tuple. */
    return Py_BuildValue ( "ddd", c, sr, sc );
}



static PyObject *
Pysobi_apply_rot ( PyObject *self, PyObject *args ) {

    double *data;
    double c, sr, sc;
    unsigned long p, q;
    unsigned long nelem, msiz;

    PyObject *input, *nparray;


    /* Gets the inputs. */
    if ( !PyArg_ParseTuple ( args, "Okkddd", &input, &p, &q, &c, &sr, &sc ) )
        return NULL;

    /* Extracts the NumPy array from the input. */
    nparray = PyArray_FROM_OTF ( input, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY );


    /* Checks the data type of the inputs. */
    if ( nparray == NULL ) {
        PyErr_SetString ( PyExc_TypeError, "Input must be a two- or three-dimensional NumPy arrays." );
        PyErr_Occurred ();
        return NULL;
    }

    /* Checks that the entry is a 2D array. */
    if ( PyArray_NDIM ( nparray ) != 2 && PyArray_NDIM ( nparray ) != 3 ) {
        PyErr_SetString ( PyExc_TypeError, "Input must be a two- or three-dimensional NumPy arrays." );
        PyErr_Occurred ();
        return NULL;
    }


    /* Gets the data size. */
    msiz  = PyArray_DIMS ( nparray ) [0];
    nelem = PyArray_DIMS ( nparray ) [ PyArray_NDIM ( nparray ) - 1 ];

    /* Gets the data itself. */
    data  = (double *) PyArray_DATA ( nparray );


    /* Applies the Givens rotation to the p and q elements in-place. */
    if ( PyArray_NDIM ( nparray ) == 2 ) {
        apply_rot_2d ( data, nelem, p, q, c, sr, sc );
    } else {
        apply_rot_3d ( data, msiz, nelem, p, q, c, sr, sc );
    }

    /* Frees the NumPy array object. */
    Py_DECREF ( nparray );


    /* Returns the updated data. */
    //return Py_BuildValue ( "O", input );

    /* Returns nothing. */
    Py_RETURN_NONE;
}



/* Defines the method table (available methods). */
static PyMethodDef
auxsobi_methods [] = {
    { "get_rot",   Pysobi_rot,  METH_VARARGS, "Calculates the rotation matrix for a pair of SOBI components." },
    { "get_svd",   Pysobi_svd,  METH_VARARGS, "Calculates the SVD of a masked 3x3 (2x2 extended) matrix." },
    { "apply_rot", Pysobi_apply_rot, METH_VARARGS, "Applies the rotation matrix to a two- or three-dimensional matrix." },
    { NULL, NULL, 0, NULL }        /* Sentinel */
};


/* Describes the module */
static struct
PyModuleDef auxsobimodule = {
    PyModuleDef_HEAD_INIT,
    "auxsobi",
    PyDoc_STR ( "Auxiliar module for SOBI in the BSS module." ),
    -1,
    auxsobi_methods
};


/* Module's initialization function. */
PyMODINIT_FUNC
PyInit_auxsobi ( void ) {

    /* Required to use NumPy's C-API. */
    import_array ();

    return PyModule_Create ( &auxsobimodule );
}


int
main ( int argc, char *argv [] ) {
    wchar_t *program = Py_DecodeLocale ( argv [0], NULL );
    if ( program == NULL ) {
        fprintf ( stderr, "Fatal error: cannot decode argv[0]\n" );
        exit (1);
    }

    /* Defines the status and the configuration. */
    PyStatus status;
    PyConfig config;
    PyConfig_InitPythonConfig ( &config );

    /* Defines the name of the program. */
    status = PyConfig_SetString ( &config, &config.program_name, program );
    if ( PyStatus_Exception ( status ) ) goto exception;

    /* Initializes the Python interpreter. */
    status = Py_InitializeFromConfig ( &config );
    if ( PyStatus_Exception ( status ) ) goto exception;

    /* Clears the configuration object from memory. */
    PyConfig_Clear ( &config );

    /* Imports the module. */
    PyObject *pmodule = PyImport_ImportModule ( "auxsobi" );

    if ( !pmodule ) {
        PyErr_Print ();
        fprintf ( stderr, "Error: could not import module 'auxsobi'\n" );
    }

    /* Clears the program from memory. */
    PyMem_RawFree ( program );
    return 0;

    /* Handles the errors during initialization. */
    exception:
    PyConfig_Clear ( &config );
    Py_ExitStatusException ( status );
}
