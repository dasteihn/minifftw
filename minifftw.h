/*
 *  Copyright 2020, Philipp Stanner, <stanner@posteo.de>
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

#ifndef __minifftw_
#define __minifftw_

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdbool.h>

#ifdef MFFTW_MPI
#include <fftw3-mpi.h>
#else
#include <fftw3.h>
#endif /* MFFTW_MPI */

#define MFFTW_REAL 0
#define MFFTW_IMAG 1

PyObject* mfftw_encapsulate_plan(fftw_plan, PyObject*, fftw_complex*, fftw_complex*);
struct mfftw_plan* mfftw_unwrap_capsule(PyObject *);

int mfftw_prepare_for_execution(struct mfftw_plan *);
int mfftw_prepare_for_output(struct mfftw_plan *);
void mfftw_data_from_npy_to_fftw(PyObject *, fftw_complex *, Py_ssize_t);
void mfftw_data_from_fftw_to_npy(PyObject *, fftw_complex *, Py_ssize_t);
long long check_array_and_get_length(PyObject *);

char** check_get_str_array(PyObject *, int);


struct mfftw_plan {
	Py_ssize_t data_len;
	PyObject *orig_arr;
	fftw_plan plan;
	fftw_complex *input_arr, *output_arr;
};


#endif /* __minifftw_ */
