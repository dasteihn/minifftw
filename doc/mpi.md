### Python and the Message Passing Interface

There is one major problem: Programs using MPI *should* be compiled using an
MPI-compiler, which takes care of correctly linking your program with your
targets MPI libraries.

'Should' means that the MPI-C-Compiler becomes most necessary when the MPI libraries
and headers are not treated as every other C-library on your system.

While you can command the python setup utility to build
with a different compiler (i.e. mpicc), the python toolchain will always ignore
the last buildstep in which the shared object will be generated.

The reason is, apparently, that the toolchain tries to ensure that each
extension is always build with the same compiler and compiler configuration
as the python interpreter was, to guarantee binary compatiblity.

**What does this mean for me, the enduser?**

If your system deploys everything needed for MPI in the standard UNIX paths such
as usr/lib: Nothing.

If your system is, however, for example a Linux cluster with a very customized
module and library system, you might have to make adjustements. This repo so far
'solves' the problem by using the normal python toolchain to build the first
parts of the extension. Then, the actual python extension is build by feeding
the hardcoded compiler flags of the last step of the python toolchain to the
cluster's mpi-c-compiler.

This solution is far from perfect, but worked for the creator of this repo. If
you have a better solution, your patch will be very welcome.
