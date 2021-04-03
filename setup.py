from distutils.core import setup, Extension
import os
import numpy

module_normal = Extension('minifftw',
        include_dirs = [numpy.get_include()],
        sources = ['minifftw.c', 'util.c', 'plancapsule.c'],
        libraries = ['fftw3'],
        extra_compile_args = ['-pthread', '-lfftw3_threads', '-lfftw3', '-lm'],
        extra_link_args = ['-lpthread', '-lfftw3_threads'],
        )

module_mpi = Extension('minifftw',
        include_dirs = [numpy.get_include()],
        sources = ['minifftw.c', 'util.c', 'plancapsule.c'],
        libraries = ['fftw3'],
        extra_compile_args = ['-pthread', '-lfftw3_mpi', '-lfftw3_threads',
            '-lm', '-D MFFTW_MPI'],
        extra_link_args = ['-lpthread', '-lfftw3_threads', '-lfftw3_mpi'],
        )


# Primitve, but works. Got a problem with it? Then improve the distutils docu. 
mpi_env = int(os.getenv("MFFTW_MPI", 0))
if mpi_env == 1:
    print("building with mpi...")
    os.environ['CC'] = "mpicc"
    main_module = module_mpi
else:
    print("building without mpi...")
    main_module = module_normal

setup(name = 'minifftw', version = '0.9',
        license = 'GPLv3',
        description = 'minimalistic, incomplete FFTW wrapper without MPI support.',
        ext_modules = [main_module],
        author='Philipp Stanner',
        author_email='stanner@posteo.de')
