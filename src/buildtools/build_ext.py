from setuptools.command.build_ext import build_ext
import numpy

class BuildExt(build_ext):
    def finalize_options(self):
        super().finalize_options()
        np_inc = numpy.get_include()
        # Append NumPy headers to every extension
        for ext in self.extensions:
            if np_inc not in ext.include_dirs:
                ext.include_dirs.append(np_inc)
