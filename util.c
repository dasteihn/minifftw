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
 * along with Minifftw.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <Python.h>
#include <stdbool.h>
#include <stdio.h>


bool
is_complex_list(PyObject *o)
{
	PyObject *first_item = PyList_GetItem(o, 0);
	return (bool)PyComplex_Check(first_item);
}


static void
fill_array(PyObject *list, Py_complex *array, Py_ssize_t total_len)
{
	puts("entered fill array");
	size_t i;
	PyObject *iterator = PyObject_GetIter(list);
	if (!iterator || PyIter_Check(iterator) == 0) {
		puts("is not iterable");
		return;
	}
	puts("is iterable");
	PyObject *iter_obj = NULL;

	for (i = 0; (iter_obj = PyIter_Next(iterator)) && i <= total_len; i++) {
		printf("looping %li\n", i);
		array[i] = PyComplex_AsCComplex(iter_obj);
		Py_DECREF(iter_obj);
	}

	Py_DECREF(iterator);
}


Py_complex *
complex_list_to_c_array(PyObject *list)
{
	Py_ssize_t list_len = PyList_Size(list);
	Py_complex *array = calloc(list_len, sizeof(Py_complex));
	if (!array) {
		perror("complex_list_to_c_array");
		return NULL;
	}

	fill_array(list, array, list_len);

	return array;
}


int
fill_fftw_array(PyObject *list, fftw_complex *array, Py_ssize_t total_len)
{
	puts("entered fill array");
	size_t i;
	PyComplex tmp = {0};
	PyObject *iter_obj = NULL;
	PyObject *iterator = PyObject_GetIter(list);
	if (!iterator || PyIter_Check(iterator) == 0) {
		puts("is not iterable");
		return -1;
	}
	puts("is iterable");

	for (i = 0; (iter_obj = PyIter_Next(iterator)) && i < total_len; i++) {
		printf("looping %li\n", i);
		tmp = PyComplex_AsCComplex(iter_obj);
		array[MFFTW_REAL] = tmp[MFFTW_REAL];
		array[MFFTW_IMAG] = tmp[MFFTW_IMAG];
		Py_DECREF(iter_obj);
	}

	Py_DECREF(iterator);
	return 0;
}
