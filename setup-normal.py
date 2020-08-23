from distutils.core import setup, Extension
import os

module1 = Extension('minifftw',
        sources = ['minifftw.c', 'util.c', 'plancapsule.c'],
        libraries = ['fftw3'],
        extra_compile_args = ['-pthread', '-lfftw3_threads', '-lfftw3', '-lm'],
        extra_link_args = ['-lpthread', '-lfftw3_threads'],
        )

# Use the mpi-C-Compiler-Wrapper. Setting the environment variable from extern
# causes awkward behavior.
# os.environ['CC'] = "mpicc"

setup(name = 'minifftw', version = '0.1',
        description = 'minimalistic, uncomplete FFTW wrapper without MPI support.',
        ext_modules = [module1],
        author='Philipp Stanner',
        author_email='stanner@posteo.de')
