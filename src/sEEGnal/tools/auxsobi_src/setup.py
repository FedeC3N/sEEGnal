from setuptools import setup, Extension
import numpy

module = Extension(
    "auxsobi",             # Must match PyInit_mymodule name
    sources=["auxsobi.c"],
    include_dirs=[numpy.get_include()]# Your C source file(s)
)

setup(
    name="auxsobi",
    version="1.0",
    description="Auxiliar module for SOBI in the BSS module.",
    ext_modules=[module],
)
