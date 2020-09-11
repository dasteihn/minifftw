# Minifftw on the Leibniz Rechenzentrum (LRZ)

Naturally, you'll want to check out the official LRZ documentation at:

https://doku.lrz.de/display/PUBLIC/High+Performance+Computing

For Minifftw it is important that you keep in mind that including the right
modules on LRZ systems might work quite differently depending on which cluster
you build. For example, at the time this repo has been created, the fftw-mpi
module on CoolMUC-3 was named `fftw/3.3.8-intel-impi`, whereas it is called
`fftw/3.3.8-intel19-impi` on CoolMUC-2.

Also, keep an eye on the LRZ's environment variables which may give important
information to you about the process of compiling and linking.

Changes on LRZ in the future may break this build-script, but even then it should
give you a rough idea how to get things to work ;)


## Build

> **Important Note:** You should call `make clean` every time before you rebuild.
This repo's setup-scripts are not very good at detecting changes.

### Prepare a Python environment

You will need a virtual python environment which contains some central utils
like distutils, numpy and so on.

TODO

### CoolMUC-2

To build on this cluster, run:

```
module load fftw/3.3.8-intel19-impi
module load python/3.6_intel
source activate YOUR-PYTHON-ENV
```

Then, from the main folder of this repo, just run `make lrz-mpi`, or `make lrz`
if you happen to want to build without MPI support.

You'll get an importable python shared object in the build/ folder. Import it
according your own design into your python programs.
