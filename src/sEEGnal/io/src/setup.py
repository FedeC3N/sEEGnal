from setuptools import setup, Extension

module = Extension(
    "raweep",             # Must match PyInit_mymodule name
    sources=["raweep.c"],
)

setup(
    name="raweep",
    version="1.0",
    description="Based on the description of the EEP 3.x file format in: cnt_riff.txt by Rainer Nowagk & Maren Grigutsch.",
    ext_modules=[module],
)
