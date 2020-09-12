# Special setup script for usage on the Leibniz Rechenzentrum Linux Clusters

from distutils.core import setup, Extension
import os
import numpy

maindir = os.environ("MFFTW_BASE")
sourcefiles = []
sourcefiles.append(maindir + "/minifftw.c")
sourcefiles.append(maindir + "/plancapsule.c")
sourcefiles.append(maindir + "/util.c")

module_normal = Extension('minifftw',
        sources = sourcefiles,
        include_dirs = [numpy.get_include()],
        libraries = ['fftw3'],
        extra_compile_args = [os.environ["FFTW_INC"], '-pthread', '-lfftw3_threads',
            '-lfftw3', '-lm'],
        extra_link_args = [os.environ["FFTW_INC"], '-lpthread', '-lfftw3_threads'],
        )

module_mpi = Extension('minifftw',
        sources = sourcefiles,
        include_dirs = [numpy.get_include()],
        libraries = ['fftw3'],
        extra_compile_args = [os.environ["FFTW_INC"], os.environ["FFTW_SHLIB"],
            os.environ.get("FFTW_MPI_LIB"), os.environ.get("MPI_LIB"),
            '-pthread', '-lfftw3_mpi', '-lfftw3_threads',
            '-lm', '-D MFFTW_MPI'],
        extra_link_args = [os.environ["FFTW_INC"], '-lpthread', '-lfftw3_threads',
            '-lfftw3_mpi'],
        )


# Primitve, but works. Got a problem with it? Then improve the distutils docu. 
mpi_env = int(os.getenv("MFFTW_MPI", 0))
if mpi_env == 1:
    print("building with mpi...")
    os.environ['CC'] = "mpiicc"
    main_module = module_mpi
else:
    print("building without mpi...")
    main_module = module_normal

setup(name = 'minifftw', version = '0.2',
        description = 'minimalistic, uncomplete FFTW wrapper without MPI support.',
        ext_modules = [main_module],
        author='Philipp Stanner',
        author_email='stanner@posteo.de')
