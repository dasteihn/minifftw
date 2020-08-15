from distutils.core import setup, Extension

module1 = Extension('minifftw', sources = ['extension.c'])

setup(name = 'Spami', version = '1.0', description = 'schrott',
        ext_modules = [module1])
