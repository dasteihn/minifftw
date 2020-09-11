# Minimalistic MPI-FFTW

This is a minimalistic Pythonwrapper for the MPI-FFTW. As every human
achievement, it is created out of hate for existing solutions.

Minimalistic means, that this wrapper will not wrapp every functionality the
FFTW offers. Probably just the 1D transforms.

This wrapper is tied to numpy and only accepts numpy arrays as payload.

## Project Status

This project is work in progress and currently in late alpha state.
It is usable, though stability and performance can not be guaranteed.

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
module and library system, you 

## Requirements

On your system, you'll need the following components:

- fftw3 C-library with header files
- fftw3\_mpi C-library with header files
- MPI implementation (i.e. openMPI) with header files

OpenMP (without 'I') is **not** required, as this wrapper uses the FFTW with POSIX threads.

## Building

- `make normal` for a wrapper around the serial FFTW
- `make mpi` for a wrapper around the MPI-FFTW, usable i.e. on clusters


## Usage

> See in `/tests` for examples.

### Basics

``` Python3
import sys
import numpy as np
import minifftw as mfftw

nr_of_threads = 8

mfftw.init(sys.argv, nr_of_threads)
data = np.random.random(2048) + np.random.random(2048) * 1j
plan = mfftw.plan_dft_1d(data, m.FFTW_FORWARD, m.FFTW_PATIENT)
result = mfftw.execute(plan)

# ...
mfftw.finit()
```

The first call (mfftw.init) is very important when using MPI: It will take your
environment and pass it to the MPI\_Init() function. Also, this function will
configure the number of threads the FFTW uses to heat up your machine.

The plan function will prepare everything the FFTW needs to operate and pack
it into a python-capsule, which later has to be passed to all following
mfftw functions.

Once you are done transforming everything you wanted, call the finit() function
to terminate MPI properly.

> Note that currently the MPI-version will terminate your whole program when you
call finit(). This is a workaround solution for some (not very well understood...)
problem terminating the whole MPI suite.

> **Note**: You can and *should* call init() and finit() regardless wethetr you
use the MPI version or not. This way, you will never have to adjust your python
code when using this wrapper, even when you'll run it on a cluster.

### Important Notes

This wrapper applies a few tricks to make the usage of FFTW more easy for the
end user. This results in a few points which should be kept in mind:

#### Plan Capsules

`mfftw.plan_XXX` returns a python capsule (opaque data) which encapsulates
three arrays: Your original numpy array and two fftw\_complex arrays for the 
FFTW to operate on.

Note the following:

- Once the plan is created, you must not reallocate your numpy-aray, nor change
its length. Violating this rule might result in undefined behavior.
- A plan and the underlying memory gets freed once the plan gets out of scope
(rather: is garbage collected). You could enforce this with `del(plan)`.


#### Executing

Executing will overwrite your original numpy array.
If you don't want this behavior, you have to pass a deep copy of your array to
the plan-method. However, if you would want to execute the plan a second time,
this would mean you'd have to copy the results of the transform into the
deep-copied first array.

## TODO

- mmap: Use mmap instead of memcpy() to speed up data exchange between python
and C.
- save memory 1: Currently, creating a plan to transform 1GB of data will result
in two additional gigabytes of RAM usage. This is partly because I'm lazy, but
also because it allows FFTW to use the DESTROY\_INPUT mode, which is often
benefitial to performance.
- save memory 2: Look into the FFTW's possibilities for MPI data distribution
- Think about exposing more of the API to the user, especially more transforms
- Wisdom: Implement the FFTW's wisdom functionality

## License

 (C) 2020, Philipp Stanner, `<stanner@posteo.de>`

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program. If not, see `<http://www.gnu.org/licenses/>`.

