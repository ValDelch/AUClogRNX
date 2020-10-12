from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
    name = 'AUClogRNX',
    ext_modules=[
        Extension('AUClogRNX_cython',
                  sources=['AUClogRNX_cython.pyx'],
                  extra_compile_args=['-O3'],
                  language='c++'),
        Extension('AUClogRNX_NN_cython',
                  sources=['AUClogRNX_NN_cython.pyx'],
                  extra_compile_args=['-O3'],
                  language='c++')
        ],
    include_dirs=[np.get_include()],
    cmdclass = {'build_ext': build_ext}
)
