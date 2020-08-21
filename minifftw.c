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

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <fftw3.h>
#include <stdio.h>
#include <stdbool.h>

#include "minifftw.h"

static PyObject *Mfftw_error = NULL;


static int
allocate_arrays(unsigned long long len, fftw_complex **in_arr, fftw_complex **out_arr)
{
	*in_arr = calloc(len, sizeof(fftw_complex));
	if (!*in_arr)
		return -1;

	*out_arr = calloc(len, sizeof(fftw_complex));
	if (!*out_arr) {
		free(*in_arr);
		return -1;
	}

	return 0;
}

// fftw_plan_dft_1d(NUM_POINTS, result, result, FFTW_FORWARD, FFTW_ESTIMATE);
static PyObject*
plan_dft_1d(PyObject *self, PyObject *args)
{
	PyObject *list = NULL;
	fftw_plan plan;
	fftw_complex *input_array = NULL, *output_array = NULL;
	unsigned long long list_len = 0;
	int direction, flags;
	int ret = PyArg_ParseTuple(args, "O!ii", &PyList_Type, &list,
			&direction, &flags);
	puts("alive 1");
	if (!list) return NULL;
	puts("alive 2");
	if (ret == 0) return NULL;

	if (PyList_Check(list) == 0) {
		PyErr_SetString(PyExc_TypeError, "Expected a list of complex numbers.");
		return NULL;
	} else {
		list_len = (unsigned long long)PyList_Size(list);
		printf("The list is %llu long.\n", list_len);
	}

	if (!is_complex_list(list)) {
		PyErr_SetString(PyExc_TypeError, "Expected a list of complex numbers.");
		return NULL;
	}
	list_len = PyList_Size(list);

	if (allocate_arrays(list_len, &input_array, &output_array) != 0)
		return PyErr_NoMemory();

	plan = fftw_plan_dft_1d(list_len, input_array, output_array, direction, flags);
	return mfftw_encapsulate_plan(plan, list, input_array, output_array);
}


static PyObject*
parse_complex(PyObject *self, PyObject *args)
{
	Py_ssize_t list_len = 0;
	PyObject *list = NULL;
	Py_complex *array = NULL;
	puts("alive 0");
/* 
 * TODO:
 * We need to be able to parse the data as np-arrays, not python-lists.
 */
	int ret = PyArg_ParseTuple(args, "O!", &PyList_Type, &list);
	puts("alive 1");
	if (!list) return NULL;
	puts("alive 2");
	if (ret == 0) return NULL;

	if (PyList_Check(list) == 0)
		puts("not a list :(");
	else {
		list_len = PyList_Size(list);
		printf("The list is %lu long.\n", list_len);
	}

	if (is_complex_list(list))
		puts("List is complex.");
	else {
		puts("list is not complex.");
		return Py_None;
	}

	array = complex_list_to_c_array(list);
	puts("alive 3");
	printf("first imag element: %lf\n", array[0].imag);
	free(array);
	array = NULL;

	return Py_None;
}



static PyMethodDef Minifftw_methods[] = {
	{"parse_complex", parse_complex, METH_VARARGS, "Build stuff from bytes"},
	{"plan_dft_1d", plan_dft_1d, METH_VARARGS, "one dimensional FFTW"},
	{NULL, NULL, 0, NULL},
};


static struct PyModuleDef fftwmodule = {
	PyModuleDef_HEAD_INIT,
	"minifftw",
	NULL,
	-1, /* awkward interpreter state foo */
	Minifftw_methods,
};

	
PyMODINIT_FUNC
PyInit_minifftw(void)
{
	PyObject *m;

	m = PyModule_Create(&fftwmodule);
	if (!m)
		return NULL;

	Mfftw_error = PyErr_NewException("spam.error", NULL, NULL);
	Py_XINCREF(Mfftw_error);
	if (PyModule_AddObject(m, "error", Mfftw_error) < 0) {
		Py_XDECREF(Mfftw_error);
		Py_CLEAR(Mfftw_error);
		Py_DECREF(m);
		return NULL;
	}
	PyModule_AddIntMacro(m, FFTW_FORWARD);
	PyModule_AddIntMacro(m, FFTW_BACKWARD);
	PyModule_AddIntMacro(m, FFTW_ESTIMATE);

	return m;
}
