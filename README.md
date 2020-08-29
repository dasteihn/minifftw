# Minimalistic MPI-FFTW

This is a minimalistic Pythonwrapper for the MPI-FFTW. As every human
achievement, it is created out of hate for existing solutions.

Minimalistic means, that this wrapper will not wrapp every functionality the
FFTW offers. Probably just the 1D transforms.

This wrapper is tied to numpy an only accepts numpy arrays as payload.

## Project Status

This project is work in progress and currently in late alpha state.
It is usable, though stability and performance can not be guaranteed.

## Requirements

On your system, you'll need the following components:

- fftw3 C-library with header files
- fftw3\_mpi with header files
- MPI implementation with header files

## Building

- `make normal` for a wrapper around the serial FFTW
- `make mpi` for a wrapper around the MPI-FFTW, usable i.e. on clusters

## Usage

> See in `/tests` for examples.

### Basics

``` Python3
import numpy as np
import minifftw as mfftw

data = np.random.random(2048) + np.random.random(2048) * 1j
plan = mfftw.plan_dft_1d(data, m.FFTW_FORWARD, m.FFTW_PATIENT)
result = mfftw.execute(plan)
```

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

