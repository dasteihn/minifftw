# Minimalistic MPI-FFTW

This is a minimalistic Pythonwrapper for the MPI-FFTW. As every human
achievement, it is created out of hate for existing solutions.

Minimalistic means, that this wrapper will not wrapp every functionality the
FFTW offers. Probably just the 1D transforms.

This wrapper is tied to numpy and only accepts numpy arrays as payload.

Please note that due to a rather complicated situation with the Message Passing
Interface, minifftw does not (yet) support real distributed memory. So if you're
working on a cluster consisting of nodes with e.g. 40GB *available* RAM per node,
the largest array you'll be able to transform is 40GB large. Have a look at
[mpi.md](./doc/mpi.md) for details.

Also note that, just like the C-FFTW, this wrapper initializes and finalizes
the Message Passing Interface automatically, without giving you direct control
over it.

![minifftw callstack](doc/images/fftw-calls.png)

Minifftw is a wrapper either around the fftw3, or fftw3-mpi, depending on how
you build it.


## Project Status

This project is work in progress and currently in beta state.
It should be usable and free of errors.

### Implemented FFTW functionality

- fftw\_init\_threads
- fftw\_init, with and without MPI
- fftw\_plan\_dft\_1d
- fftw cleanup routines


## List of supported Linux Clusters

- [Leibniz Rechenzentrum](./clusters/lrz/README.md), CoolMUC-2

Please send a patch if you want to see your cluster supported.


## Requirements

On your system, you'll need the following components:

- fftw3 C-library with header files
- fftw3\_mpi C-library with header files
- Numpy C-header files
- MPI implementation (i.e. openMPI) with header files

OpenMP (without 'I') is **not** required, as this wrapper uses the FFTW with
POSIX threads.

On a typical linux distro, the required packages might be called:

- libfftw3-mpi-dev
- python3-numpy
- libopenmpi-dev


## Building

### Conventional Targets

- `make normal` for a wrapper around the serial FFTW
- `make mpi` for a wrapper around the MPI-FFTW

### Linux Clusters

See in clusters/ for a list of the supported clusters. The folder also contains
sub-READMEs which are customized to the cluster and will hopefully help you
getting the wrapper to run on your target.

To build, run from the main folder:

`make <cluster>` or `make <cluster>-mpi`


## Usage

> See in `tests/` for examples.

### Basics

``` Python3
import sys
import numpy as np
import minifftw as mfftw

nr_of_threads = 8
data_len = 2048

mfftw.init(sys.argv, nr_of_threads)
data_in = np.random.random(data_len) + np.random.random(data_len) * 1j
data_out = np.zeros(data_len, dtype="complex128")
p = m.plan_dft_1d(data_in, data_out, m.FFTW_FORWARD, m.FFTW_ESTIMATE)

# the assignment is optional. mfftw.execute will fill data_out automatically
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
to terminate MPI properly. When building the MPI-version, finit() will terminate
*all* your processes. So, make sure you saved everything before this call.

> **Note**: You can and *should* call init() and finit() regardless whether you
use the MPI version or not. This way, you will never have to adjust your python
code when using this wrapper, even when you'll run it on a cluster.

### Important Notes

This wrapper applies a few tricks to make the usage of FFTW more easy for the
end user. This results in a few points which should be kept in mind.

Additionally, keep in mind that minifftw just wraps FFTW – so the behavior
documented for the FFTW will also apply to this wrapper. For instance, the
planner-functions will overwrite your input array with arbitrary data, so you
should fill them with your payload once planing is completed.

#### Plan Capsules

`mfftw.plan_XXX` returns a python capsule (opaque data) which encapsulates
the fftw data types, including the fftw\_plan and a reference to your numpy arrays.

Note the following:

- Once the plan is created, you must not reallocate your numpy-array, nor change
its length. Violating this rule might result in undefined behavior.
- A plan and the underlying memory (except if there are other references to the
numpy arrays) gets freed once the plan gets out of scope
(rather: is garbage collected). You could enforce this with `del(my_plan)`.
- The capsule contains a reference to your numpy-array. Therefore, the array can
not be garbage collected until the plan-capsule gets dropped.

#### Inplace Transforms and Overwriting

If input\_array and output\_array are identical, the wrapper resp. the fftw will
perform an inplace transform, hence overwriting your original array.

Additionally, FFTW offers you the opportunity to pass two different arrays, but
allow the library to overwrite your input-array anyways. This might help the
FFTW gain performance. To enable this mode, pass `minifftw.FFTW_DESTROY_INPUT`
as a flag in the plan creation functions.


## TODO

- Think about exposing more of the API to the user, especially more transforms
- Wisdom: Implement the FFTW's wisdom functionality
- Distributed Memory: This version does not yet support the functionality for
real distributed memory. This means that the FFTW's `alloc_local` functions are
not yet available. Implementing this will be especially useful when the
multidimensional transforms get implemented.

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

