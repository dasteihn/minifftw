from distutils.core import setup, Extension
import os
import sys

os.environ["CC"] = "mpicc"

module1 = Extension('minifftw',
        sources = ['minifftw.c', 'util.c', 'plancapsule.c'],
        libraries = ['fftw3'],
        extra_compile_args = ['-pthread', '-lfftw3_mpi', '-lfftw3_threads',
            '-lm', '-D MFFTW_MPI'],
        extra_link_args = ['-lpthread', '-lfftw3_threads', '-lfftw3_mpi'],
        )

setup(name = 'minifftw', version = '0.2',
        description = 'minimalistic, uncomplete FFTW wrapper with MPI support.',
        ext_modules = [module1],
        author='Philipp Stanner',
        author_email='stanner@posteo.de')
