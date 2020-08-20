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

bool is_complex_list(PyObject *);
Py_complex* complex_list_to_c_array(PyObject *);

struct mini_fftw_plan {
	PyObject *original_list;
	fftw_complex *input_array;
	fftw_plan plan;
};


#endif /* __minifftw_ */
