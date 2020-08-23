from distutils.core import setup, Extension
import os

module1 = Extension('minifftw',
        sources = ['minifftw.c', 'util.c', 'plancapsule.c'],
        libraries = ['fftw3'],
        extra_compile_args = ['-pthread', '-lfftw3_mpi', '-lfftw3_threads',
            '-lm', '-D MFFTW_MPI'],
        extra_link_args = ['-lpthread', '-lfftw3_threads', '-lfftw3_mpi'],
        )

# Use the mpi-C-Compiler-Wrapper. Setting the environment variable from extern
# causes awkward behavior.
os.environ['CC'] = "mpicc"

setup(name = 'minifftw', version = '0.1', description = 'schrott',
        ext_modules = [module1],
        author='Philipp Stanner',
        author_email='stanner@posteo.de')
