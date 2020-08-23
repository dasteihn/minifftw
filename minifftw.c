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


static PyObject *
plan_dft_1d(PyObject *self, PyObject *args)
{
	PyObject *list = NULL;
	fftw_plan plan;
	fftw_complex *input_array = NULL, *output_array = NULL;
	unsigned long long list_len = 0;
	int direction, flags;
	int ret = PyArg_ParseTuple(args, "O!ii", &PyList_Type, &list,
			&direction, &flags);
	if (ret == 0 || !list)
		return NULL;

	if (PyList_Check(list) == 0) {
		PyErr_SetString(PyExc_TypeError, "Expected a list of complex numbers.");
		return NULL;
	}
	list_len = (unsigned long long)PyList_Size(list);

	if (!is_complex_list(list)) {
		PyErr_SetString(PyExc_TypeError, "Expected a list of complex numbers.");
		return NULL;
	}
	list_len = PyList_Size(list);

	if (allocate_arrays(list_len, &input_array, &output_array) != 0)
		return PyErr_NoMemory();

	/*
	 * currently, we allocate one array more than necessary, since we
	 * return output data directly to the python list. So we allow the FFTW
	 * to use the input array as it pleases, to be quicker (possibly).
	 */
	flags |= FFTW_DESTROY_INPUT;

#ifdef MFFTW_MPI
	plan = fftw_mpi_plan_dft_1d(list_len, input_array, output_array,
			MPI_COMM_WORLD, direction, flags);
#else
	plan = fftw_plan_dft_1d(list_len, input_array, output_array, direction, flags);
#endif

	return mfftw_encapsulate_plan(plan, list, input_array, output_array);
}


static PyObject *
execute(PyObject *self, PyObject *args)
{
	struct mfftw_plan *mplan = NULL;
	PyObject *plancapsule = NULL;
	int ret = PyArg_ParseTuple(args, "O", &plancapsule);
	if (ret == 0 || !plancapsule)
		return NULL;
	mplan = mfftw_unwrap_capsule(plancapsule);
	if (!mplan)
		return NULL;

	if (mfftw_prepare_for_execution(mplan) != 0)
		return NULL;

	fftw_execute(mplan->plan);
	if (mfftw_prepare_for_output(mplan) != 0) {
		puts("getting data back failed.");
		return NULL;
	}

	return mplan->orig_list;
}


static bool
initialize_threaded_mpi(PyObject *argv_list)
{
	int passed_argc = 0, provided = 0;
	char **passed_argv = NULL;
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
	passed_argv = check_get_str_array(argv_list, passed_argc);
	if (!passed_argv)
		return false;

	/*
	 * FUNNELED means: Only the main thread will make MPI-calls
	 */
	MPI_Init_thread(&passed_argc, &passed_argv, MPI_THREAD_FUNNELED, &provided);
	free(passed_argv);

	return (bool)(provided >= MPI_THREAD_FUNNELED);
}


static PyObject *
init(PyObject *self, PyObject *args)
{
	bool threads_ok = true;
	int nr_of_threads = 4;
	PyObject *argv_list = NULL;

	int ret = PyArg_ParseTuple(args, "O!i", &PyList_Type, &argv_list,
			&nr_of_threads);
	if (ret == 0 || !argv_list)
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

	return Py_None;
}


static PyObject *
finit(PyObject *self, PyObject *args)
{
#ifdef MFFTW_MPI
	fftw_mpi_cleanup();
//	MPI_Finalize();
#else
	fftw_cleanup();	
#endif /* MFFTW_MPI */

	return Py_None;
}


static PyMethodDef Minifftw_methods[] = {
	{"init", init, METH_VARARGS, "prepare FFTW and (if desired) MPI"},
	{"finit", finit, METH_VARARGS, "finalize everything"},
	{"plan_dft_1d", plan_dft_1d, METH_VARARGS, "one dimensional FFTW"},
	{"execute", execute, METH_VARARGS, "execute a previously created plan"},
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
	PyModule_AddIntMacro(m, FFTW_MEASURE);
	PyModule_AddIntMacro(m, FFTW_PATIENT);
	PyModule_AddIntMacro(m, FFTW_EXHAUSTIVE);
	PyModule_AddIntMacro(m, FFTW_WISDOM_ONLY);

	return m;
}
