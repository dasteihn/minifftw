/*
 *  Copyright 2020, 2021 Philipp Stanner, <stanner@posteo.de>
 *
 * This file is part of Minifftw.
 * Minifftw is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Minifftw is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with Minifftw. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef MINIFFTW
#define MINIFFTW

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdbool.h>

/*
 * Importing complex.h /before/ any fftw header enforces full binary compatiblity
 * between fftw-complex and C-complex.
 * According to the numpy docu
 * https://numpy.org/doc/stable/user/basics.types.html?highlight=complex128
 * this is identical to numpy-complex128 which allows us to easily convert
 * between the types, without iterating over huge arrays.
 */
#include <complex.h>

#ifdef MFFTW_MPI
#include <fftw3-mpi.h>
#else
#include <fftw3.h>
#endif /* MFFTW_MPI */

#define MFFTW_REAL 0
#define MFFTW_IMAG 1

struct array_meta {
	ptrdiff_t local;
	ptrdiff_t local_ni, local_i_start;
	ptrdiff_t local_no, local_o_start;
};


struct process_map {
	int nr_of_procs;
	struct mfftw_mpi_info *infos;
};


struct mfftw_mpi_info {
	int rank;
	struct array_meta arrmeta;
	fftw_complex *local_slice;
	struct process_map procmap;
};


struct mfftw_plan {
	Py_ssize_t data_len;
	fftw_plan plan;
	PyArrayObject *in_arr, *out_arr;
	struct mfftw_mpi_info *info;
};


PyObject *
mfftw_encapsulate_plan(fftw_plan, PyArrayObject*, PyArrayObject*,
		struct mfftw_mpi_info *);
struct mfftw_plan* mfftw_unwrap_capsule(PyObject *);

void mfftw_data_from_npy_to_fftw(PyArrayObject *, fftw_complex *, Py_ssize_t);
void mfftw_data_from_fftw_to_npy(PyArrayObject *, fftw_complex *, Py_ssize_t);
long long check_array_and_get_length(PyArrayObject *);
fftw_complex* reinterpret_numpy_to_fftw_arr(PyArrayObject *np);

char** check_get_str_array(PyObject *, int);


#endif /* MINIFFTW */
