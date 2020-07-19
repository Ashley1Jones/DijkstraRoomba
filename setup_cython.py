from distutils.core import setup
from Cython.Build import cythonize
import numpy
import sys

setup(
    ext_modules=cythonize("GridMean.pyx",
                          compiler_directives={'language_level': "3"},
                          annotate=True),
    include_dirs=[numpy.get_include()]
)
