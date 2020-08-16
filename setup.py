from distutils.core import setup, Extension
import os

module1 = Extension('minifftw', sources = ['extension.c', 'util.c'])

# Use the mpi-C-Compiler-Wrapper. Setting the environment variable from extern
# causes awkward behavior.
# os.environ['CC'] = "mpicc"

setup(name = 'Spami', version = '1.0', description = 'schrott',
        ext_modules = [module1])
