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

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL mfftw_ARRAY_API
#include <numpy/arrayobject.h>

#include "minifftw.h"
#include <stdbool.h>

static char *
get_str_from_object(PyObject *o)
{
	char *ret = NULL;

	if (!PyUnicode_Check(o)) {
		PyErr_SetString(PyExc_TypeError, "Could not parse argv string.");
		return NULL;
	}

	ret = (char *)PyUnicode_DATA(o);
	if (!ret)
		PyErr_SetString(PyExc_TypeError, "Could not parse argv string.");

	return ret;
}


char **
check_get_str_array(PyObject *list, int argc_passed)
{
	char **ret = NULL;
	PyObject *tmp = NULL;

	ret = malloc(argc_passed * sizeof(char *));
	if (!ret) {
		/* raises memory exception and returns NULL */
		return (char **)PyErr_NoMemory();
	}

	for (int i = 0; i < argc_passed; i++) {
		tmp = PyList_GetItem(list, i);
		if (!tmp)
			goto err_out;
		if ((ret[i] = get_str_from_object(tmp)) == NULL)
			goto err_out;
	}

	return ret;

err_out:
	free(ret);
	return NULL;
}


/*
 * ======================== Numpy Array Utility ===============================
 */

fftw_complex *
reinterpret_numpy_to_fftw_arr(PyArrayObject *np)
{
	return (fftw_complex *)PyArray_DATA(np);
}

void
mfftw_data_from_npy_to_fftw(PyArrayObject *arr_np, fftw_complex *arr_fftw,
	Py_ssize_t total_len)
{
	void *np_raw_data = PyArray_DATA(arr_np);
	/*
	 * When getting the numpy array from the python space, it is ensured
	 * that it contains complex128 data, which should always be binary
	 * identical to the fftw_complex data type.
	 */
	memcpy(arr_fftw, np_raw_data, (size_t)total_len);
}


void 
mfftw_data_from_fftw_to_npy(PyArrayObject *arr_np, fftw_complex *arr_fftw,
	Py_ssize_t total_len)
{
	void *np_raw_data = PyArray_DATA(arr_np);
	memcpy(np_raw_data, arr_fftw, (size_t)total_len);
}

/*
 * Check if the passed Object is a numpy array which suits our needs.
 * For now, MFFTW will only allow 1-dimensional arrays.
 * Returns -1 if one of our requirements is violated.
 */
long long
check_array_and_get_length(PyArrayObject *arr)
{
	if (PyArray_CheckExact(arr) == 0) {
		PyErr_SetString(PyExc_TypeError, "Expected an numpy array.");
		return -1;
	}

	if (PyArray_NDIM(arr) != 1) {
		PyErr_SetString(PyExc_TypeError,
			"Expected a 1-dimensional numpy array.");
		return -1;
	}

	if (PyArray_TYPE(arr) != NPY_COMPLEX128) {
		PyErr_SetString(PyExc_TypeError,
			"Expected an numpy array of complex128.");
		return -1;
	}

	return (long long)PyArray_SIZE(arr);
}


