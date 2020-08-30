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

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL mfftw_ARRAY_API
#include <numpy/arrayobject.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "minifftw.h"

static PyObject *Mfftw_error = NULL;

static char **Argv = NULL;


static long long
prepare_arrays(PyObject *tmp1, PyObject *tmp2,
		PyArrayObject **arr1, PyArrayObject **arr2)
{
	long long array_len1 = 0, array_len2 = 0;
	*arr1 = (PyArrayObject *)PyArray_FROM_OTF(tmp1, NPY_COMPLEX128,
			NPY_ARRAY_IN_ARRAY);
	*arr2 = (PyArrayObject *)PyArray_FROM_OTF(tmp2, NPY_COMPLEX128,
			NPY_ARRAY_IN_ARRAY);
	if (!*arr1 || !*arr2)
		return -1;

	array_len1 = check_array_and_get_length(*arr1);
	array_len2 = check_array_and_get_length(*arr2);
	if (array_len1 < 0 || array_len2 < 0)
		return -1;
	if (array_len1 != array_len2)
		return -1;

	return array_len1;
}


static PyObject *
plan_dft_1d(PyObject *self, PyObject *args)
{
	PyObject *tmp1 = NULL, *tmp2 = NULL;
	fftw_plan plan;
	PyArrayObject *py_in_arr = NULL, *py_out_arr = NULL;
	fftw_complex *mfftw_in_arr = NULL, *mfftw_out_arr = NULL;
	long long array_len = 0;
	int success = 0, direction, flags;
	success = PyArg_ParseTuple(args, "O!O!ii", &PyArray_Type, &tmp1,
		&PyArray_Type, &tmp2, &direction, &flags);

	if (success == 0)
		return NULL;

	array_len = prepare_arrays(tmp1, tmp2, &py_in_arr, &py_out_arr);
	if (array_len < 0) {
		PyErr_SetString(Mfftw_error, "Could not prepare arrays.");
		return NULL;
	}

	mfftw_in_arr = reinterpret_numpy_to_fftw_arr(py_in_arr);
	mfftw_out_arr = reinterpret_numpy_to_fftw_arr(py_out_arr);

#ifdef MFFTW_MPI
	/*
	 * COMM_WORLD means: All existing MPI-tasks will participate in calculating.
	 */
	plan = fftw_mpi_plan_dft_1d(array_len, mfftw_in_arr, mfftw_out_arr,
		MPI_COMM_WORLD, direction, flags);
#else
	plan = fftw_plan_dft_1d(array_len, mfftw_in_arr, mfftw_out_arr,
		direction, flags);
#endif

	return mfftw_encapsulate_plan(plan, py_in_arr, py_out_arr);
}


PyObject *
import_wisdom(PyObject *self, PyObject *args)
{
	char *wisdom_path = NULL;
	if (PyArg_ParseTuple(args, "s", &wisdom_path) != 0)
		return NULL;
	if (fftw_import_wisdom_from_filename(wisdom_path) != 0) {
		Mfftw_error = PyErr_NewException("minifftw.wisdomerror",
				NULL, NULL);
		PyErr_SetString(Mfftw_error, "fftw-wisdom can not be imported.");
		return NULL;
	}

	return Py_None;
}


PyObject *
export_wisdom(PyObject *self, PyObject *args)
{
	char *wisdom_path = NULL;
	if (PyArg_ParseTuple(args, "s", &wisdom_path) != 0)
		return NULL;
	if (fftw_export_wisdom_to_filename(wisdom_path) != 0) {
		Mfftw_error = PyErr_NewException("minifftw.wisdomerror",
				NULL, NULL);
		PyErr_SetString(Mfftw_error, "fftw-wisdom can not be stored.");
		return NULL;
	}

	return Py_None;
}


PyObject *
export_wisdom(PyObject *self, PyObject *args)
{
	char *wisdom_path = NULL;
	if (PyArg_ParseTuple(args, "s", &wisdom_path) != 0)
		return NULL;
	if (fftw_export_wisdom_to_filename(wisdom_path) != 0) {
		Mfftw_error = PyErr_NewException("minifftw.wisdomerror",
				NULL, NULL);
		PyErr_SetString(Mfftw_error, "fftw-wisdom can not be stored.");
		return NULL;
	}

	return Py_None;
}


static PyObject *
execute(PyObject *self, PyObject *args)
{
	struct mfftw_plan *mplan = NULL;
	PyObject *plancapsule = NULL;
	/* mfftw_unwrap will check the type */
	int success = PyArg_ParseTuple(args, "O", &plancapsule);
	if (success == 0 || !plancapsule)
		return NULL;
	mplan = mfftw_unwrap_capsule(plancapsule);
	if (!mplan)
		return NULL;

	fftw_execute(mplan->plan);

	Py_INCREF(mplan->out_arr);
	return (PyObject *)(mplan->out_arr);
}


#ifdef MFFTW_MPI
static bool
initialize_threaded_mpi(PyObject *argv_list)
{
	int passed_argc = 0, provided = 0;
	printf("argc: %i\n", passed_argc);

	/*
	 * passed_argc is correct as-is, because this function received the
	 * version /without/ the nr_of_threads integer in init()
	 */
	passed_argc = PyList_Size(argv_list);
	if (passed_argc <= 0) { 
		PyErr_SetString(PyExc_ValueError,
				"Length of argv list wrong for MPI usage.");
		return false;
	}
	printf("argc: %i\n", passed_argc);
	Argv = check_get_str_array(argv_list, passed_argc);
	if (!Argv)
		return false;

	 /* FUNNELED means: Only the main thread will make MPI-calls */
	MPI_Init_thread(&passed_argc, &Argv, MPI_THREAD_FUNNELED, &provided);

	return (bool)(provided >= MPI_THREAD_FUNNELED);
}
#endif /* MFFTW_MPI */


static PyObject *
init(PyObject *self, PyObject *args)
{
	bool threads_ok = true;
	int nr_of_threads = 4;
	PyObject *argv_list = NULL;

	int success = PyArg_ParseTuple(args, "O!i", &PyList_Type, &argv_list,
			&nr_of_threads);
	if (success == 0 || !argv_list)
		return NULL;

	if (PyList_Check(argv_list) == 0) {
		PyErr_SetString(PyExc_TypeError, "Expected a list of strings.");
		return NULL;
	}

#ifdef MFFTW_MPI
	threads_ok = initialize_threaded_mpi(argv_list);
#endif /* MFFTW_MPI */
	/* works without preprocessor, due to initialization with true */
	if (threads_ok)
		threads_ok = fftw_init_threads();

	if (!threads_ok) {
		/* TODO: error handling */
		return NULL;
	}

#ifdef MFFTW_MPI
	fftw_mpi_init();
#endif /* MFFTW_MPI */

	fftw_plan_with_nthreads(nr_of_threads);

	Py_RETURN_NONE;
}


/*
 * Will free the FFTW's ressources,
 * and especially will terminate all MPI processes.
 */
static PyObject *
finit(PyObject *self, PyObject *args)
{
	free(Argv);
	Argv = NULL;
#ifdef MFFTW_MPI
	fftw_mpi_cleanup();
	MPI_Finalize();
/*
 * Currently, there exists an awkward race-condition like problem with
 * finalizing MPI. The problem seems to disappear when you terminate the whole
 * python interpreter from the python-extension.
 * The problem seems only to exist when using MPI-fftw in python, whereas a plain
 * C-application runs just fine. This might mean that the python interpreter
 * process might play some role in the phenomenon.
 *
 * TODO: Find out if there is a better solution.
 */
	exit(EXIT_SUCCESS);
#else
	fftw_cleanup();	
#endif /* MFFTW_MPI */

	Py_RETURN_NONE;
}


static PyMethodDef Minifftw_methods[] = {
#ifdef MFFTW_MPI
	{"init", init, METH_VARARGS, "prepare FFTW and  MPI"},
#else
	{"init", init, METH_VARARGS, "prepare FFTW"},
#endif /* MFFTW_MPI */
	{"finit", finit, METH_VARARGS, "finalize everything"},
	{"plan_dft_1d", plan_dft_1d, METH_VARARGS, "one dimensional FFTW"},
	{"execute", execute, METH_VARARGS, "execute a previously created plan"},
	{"import_wisdom", import_wisdom, METH_VARARGS,
		"import the FFTW wisdom from a filename/path"},
	{"export_wisdom", export_wisdom, METH_VARARGS,
		"export the FFTW wisdom to a filename/path"},
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

	Mfftw_error = PyErr_NewException("minifftw.error", NULL, NULL);
	Py_XINCREF(Mfftw_error);
	if (PyModule_AddObject(m, "error", Mfftw_error) < 0) {
		Py_XDECREF(Mfftw_error);
		Py_CLEAR(Mfftw_error);
		Py_DECREF(m);
		return NULL;
	}
	/* Planing-rigor flags */
	PyModule_AddIntMacro(m, FFTW_FORWARD);
	PyModule_AddIntMacro(m, FFTW_BACKWARD);
	PyModule_AddIntMacro(m, FFTW_ESTIMATE);
	PyModule_AddIntMacro(m, FFTW_MEASURE);
	PyModule_AddIntMacro(m, FFTW_PATIENT);
	PyModule_AddIntMacro(m, FFTW_EXHAUSTIVE);
	PyModule_AddIntMacro(m, FFTW_WISDOM_ONLY);

	/* Algorithm restriction flags */
	PyModule_AddIntMacro(m, FFTW_DESTROY_INPUT);

	import_array();

	return m;
}
