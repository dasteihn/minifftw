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

/*
 * ========================= Environment ======================================
 *
 * The following functions serve the purpose the get the argv environment of the
 * program and parse it into a form evaluable by C.
 * This is done, because MPI requires the original environment as it is passed
 * by mpiexec.
 */

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
			"Expected an one-dimensional numpy array.");
		return -1;
	}

	if (PyArray_TYPE(arr) != NPY_COMPLEX128) {
		PyErr_SetString(PyExc_TypeError,
			"Expected an numpy array of complex128.");
		return -1;
	}

	return (long long)PyArray_SIZE(arr);
}


/* ============================== Debugging ================================= */

void
print_complex_nr(fftw_complex nr)
{
	printf("%lf + %lfj ", creal(nr), cimag(nr));
}


void
debug_fftw_array_print(fftw_complex *arr, int len)
{
	int i = 0;

	for (i = 0; i < len; i++) {
		print_complex_nr(arr[i]);
		if (i % 4 == 0)
			printf("\n");
	}
	printf("\n\n");
}


void
debug_array_print(struct mfftw_plan *mplan)
{
	int printed = 0;
	ssize_t i;
	fftw_complex *arr_in = reinterpret_numpy_to_fftw_arr(mplan->in_arr);
	fftw_complex *arr_out = reinterpret_numpy_to_fftw_arr(mplan->out_arr);

	for (i = 0; i < mplan->data_len; i++) {
		print_complex_nr(arr_in[i]);
		if (printed++ % 4 == 0)
			printf("\n");
	}
	printf("\n\n");

	for (i = 0; i < mplan->data_len; i++) {
		print_complex_nr(arr_out[i]);
		if (printed++ % 4 == 0)
			printf("\n");
	}

	printf("\n");
}
