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

#include <Python.h>
#include <fftw3.h>
#include <stdbool.h>

#define MFFTW_REAL 0
#define MFFTW_IMAG 1

bool is_complex_list(PyObject *);
Py_complex* complex_list_to_c_array(PyObject *);
PyObject* mfftw_encapsulate_plan(fftw_plan, PyObject*, fftw_complex*, fftw_complex*);
struct mini_fftw_plan* mfftw_unwrap_capsule(PyObject *);
int mfftw_prepare_for_execution(struct mini_fftw_plan *);
int fill_fftw_array(PyObject *, fftw_complex *, Py_ssize_t);
int fftw_arr_to_list(PyObject *, fftw_complex *, Py_ssize_t);

struct mini_fftw_plan {
	Py_ssize_t list_len;
	PyObject *orig_list;
	fftw_plan plan;
	fftw_complex *input_array, *output_array;
};


#endif /* __minifftw_ */
