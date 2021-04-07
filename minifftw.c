/*
 *  Copyright 2020, 2021, Philipp Stanner, <stanner@posteo.de>
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
	long long array_len1 = -1, array_len2 = -1;
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

#ifdef MFFTW_MPI

static void
cleanup_mfftw_mpi_info(struct mfftw_mpi_info *info)
{
	if (!info)
		return;

	memset(info, 0, sizeof(struct mfftw_mpi_info));
	free(info);
}


static struct mfftw_mpi_info *
prepare_mfftw_mpi_info(long long array_len, int direction, int flags)
{
	int rank, nr_of_procs;
	struct mfftw_mpi_info *info;

	info = calloc(1, sizeof(struct mfftw_mpi_info));
	if (!info)
		return NULL;

	info->arrmeta.local = fftw_mpi_local_size_1d(array_len, MPI_COMM_WORLD,
                direction, flags, &info->arrmeta.local_ni,
		&info->arrmeta.local_i_start, &info->arrmeta.local_no,
		&info->arrmeta.local_o_start);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	info->rank = rank;

	MPI_Comm_size(MPI_COMM_WORLD, &nr_of_procs);
	info->procmap.nr_of_procs = nr_of_procs;

	return info;
}


static int
receive_subinfo(struct array_meta *meta, int rank)
{
	int ret;
	MPI_Status stat;
	unsigned long long arrmeta[5] = {0};

	ret = MPI_Recv(arrmeta, 5, MPI_UNSIGNED_LONG_LONG, rank, 0,
			MPI_COMM_WORLD, &stat);

	meta->local = arrmeta[0];
	meta->local_ni = arrmeta[1];
	meta->local_i_start = arrmeta[2];
	meta->local_no = arrmeta[3];
	meta->local_o_start = arrmeta[4];

	return ret;
}


static int
collect_mfftw_mpi_infos(struct mfftw_mpi_info *info)
{
	int rank, ret;
	struct mfftw_mpi_info *tmp;

	tmp = calloc(info->procmap.nr_of_procs, sizeof(struct mfftw_mpi_info));
	if (!tmp)
		return -1;

	/* This is us, process 0 */
	memcpy(&tmp[0], info, sizeof(struct mfftw_mpi_info));

	for (rank = 1; rank < info->procmap.nr_of_procs; rank++) {
		ret = receive_subinfo(&tmp[rank].arrmeta, rank);
		if (ret != 0)
			break;
	}

	info->procmap.infos = tmp;

	return ret;
}


/* inform Lehnsherr */
static int
send_mfftw_mpi_info(struct array_meta *meta)
{
	/* MPI can't send ptrdiffs, so we hack... */
	unsigned long long arrmeta[5];

	arrmeta[0] = meta->local;
	arrmeta[1] = meta->local_ni;
	arrmeta[2] = meta->local_i_start;
	arrmeta[3] = meta->local_no;
	arrmeta[4] = meta->local_o_start;

	return MPI_Send(arrmeta, 5, MPI_UNSIGNED_LONG_LONG, 0, 0, MPI_COMM_WORLD);
}


/* 
 * Inform the meisterprocess about who has which slice of the array.
 * ._.
 */
static int
synchronize_process_map(struct mfftw_mpi_info *info)
{
	int ret;

	if (info->rank == 0)
		ret = collect_mfftw_mpi_infos(info);
	else
		ret = send_mfftw_mpi_info(&info->arrmeta);

	return ret;
}


/*
 * This function will do several things:
 * 1. Create the MPI FFTW plans
 * 2. Allocate the local slice the FFTW wants.
 * 3. Inform the meisterprocess about who has which slice of the whole array.
 * 4. Confuse programmers
 */
static PyObject *
plan_dft_1d_mpi(PyObject *self, PyObject *args)
{
	struct mfftw_mpi_info *info = NULL;
	PyObject *tmp1 = NULL, *tmp2 = NULL;
	fftw_plan plan;
	fftw_complex *mfftw_in_arr = NULL, *mfftw_out_arr = NULL;
	PyArrayObject *py_in_arr = NULL, *py_out_arr = NULL;
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
	
	info = prepare_mfftw_mpi_info(array_len, direction, flags);
	if (!info) {
		PyErr_SetString(Mfftw_error, "Could not prepare local info.");
		return NULL;
	}

	if (synchronize_process_map(info) != 0) {
		cleanup_mfftw_mpi_info(info);
		PyErr_SetString(Mfftw_error, "Could not synchronize MPI map.");
		return NULL;
	}

	mfftw_in_arr = reinterpret_numpy_to_fftw_arr(py_in_arr);
	mfftw_out_arr = reinterpret_numpy_to_fftw_arr(py_out_arr);

	/*
	 * COMM_WORLD means: All existing MPI-tasks will participate in calculating.
	 */
	plan = fftw_mpi_plan_dft_1d(array_len,
			mfftw_in_arr, mfftw_out_arr,
			MPI_COMM_WORLD, direction, flags);

	if (!plan) {
		cleanup_mfftw_mpi_info(info);
		PyErr_SetString(Mfftw_error, "Could not create plan.");
		return NULL;
	}

	return mfftw_encapsulate_plan(plan, py_in_arr, py_out_arr, info);
}

#else 

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

	plan = fftw_plan_dft_1d(array_len, mfftw_in_arr, mfftw_out_arr,
		direction, flags);

	if (!plan) {
		PyErr_SetString(Mfftw_error, "Could not create plan.");
		return NULL;
	}

	/* info is passed empty and is not used without MPI. */
	return mfftw_encapsulate_plan(plan, py_in_arr, py_out_arr, NULL);
}
#endif /* MFFTW_MPI */


#ifdef MFFTW_MPI

static PyObject *
get_mpi_rank(PyObject *self, PyObject *args)
{
	int rank = -1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	return Py_BuildValue("i", rank);
}


static PyObject *
import_wisdom_mpi(PyObject *self, PyObject *args)
{    
	char *wisdom_path = NULL;
	if (PyArg_ParseTuple(args, "s", &wisdom_path) == 0)
		return NULL;

	int rank = -1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) {
		if (fftw_import_wisdom_from_filename(wisdom_path) != 1) {
			/*
			PyErr_SetString(Mfftw_error, "fftw-wisdom can not be imported.");
			return NULL;
			*/
			// FIXME: warn the user without deadlock danger.
		}
	}

	fftw_mpi_broadcast_wisdom(MPI_COMM_WORLD);

	Py_RETURN_NONE;
}

static PyObject *
import_system_wisdom_mpi(PyObject *self, PyObject *args)
{
	int rank = -1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) {
		// FIXME: warn user in case of failure without deadlock danger
		fftw_import_system_wisdom();
	}

	fftw_mpi_broadcast_wisdom(MPI_COMM_WORLD);

	Py_RETURN_NONE;
}

#else

static PyObject *
get_pseudo_rank(PyObject *self, PyObject *args)
{
	int rank = 0;

	return Py_BuildValue("i", rank);
}

static PyObject *
import_wisdom(PyObject *self, PyObject *args)
{
	char *wisdom_path = NULL;
	if (PyArg_ParseTuple(args, "s", &wisdom_path) == 0)
		return NULL;

	/* fftw uses 0 as error code */
	if (fftw_import_wisdom_from_filename(wisdom_path) == 0) {
		PyErr_SetString(Mfftw_error, "fftw-wisdom can not be imported.");
		return NULL;
	}

	Py_RETURN_NONE;
}

static PyObject *
import_system_wisdom(PyObject *self, PyObject *args)
{
	if (fftw_import_system_wisdom() == 0) {
		PyErr_SetString(Mfftw_error, "Can not import system-wisdom.");
		return NULL;
	}

	Py_RETURN_NONE;
}
#endif /* MFFTW_MPI */



#ifdef MFFTW_MPI
static PyObject *
export_wisdom_mpi(PyObject *self, PyObject *args)
{
	int rank = -1, ret = 1;
	char *wisdom_path = NULL;
	if (PyArg_ParseTuple(args, "s", &wisdom_path) == 0)
		return NULL;

	fftw_mpi_gather_wisdom(MPI_COMM_WORLD);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0) {
		ret = fftw_export_wisdom_to_filename(wisdom_path);
		if (ret == 0) {
			/*
			PyErr_SetString(Mfftw_error, "Could not store mpi-fftw-wisdom.");
			return NULL;
			*/
			// FIXME: warn the user without deadlock danger
		}
	}

	Py_RETURN_NONE;
}

#else

static PyObject *
export_wisdom(PyObject *self, PyObject *args)
{
	char *wisdom_path = NULL;
	if (PyArg_ParseTuple(args, "s", &wisdom_path) == 0)
		return NULL;

	if (fftw_export_wisdom_to_filename(wisdom_path) == 0) {
		PyErr_SetString(Mfftw_error, "fftw-wisdom can not be exported.");
		return NULL;
	}

	Py_RETURN_NONE;
}

#endif /* MFFTW_MPI */

#ifdef MFFTW_MPI

static int
transmit_payload(int rank, fftw_complex *arr, size_t size)
{
	return MPI_Send(arr, size, MPI_C_DOUBLE_COMPLEX, rank, 0, MPI_COMM_WORLD);
}


static int
receive_payload(int rank, fftw_complex *arr, size_t size)
{
	MPI_Status stat;

	return MPI_Recv(arr, size, MPI_C_DOUBLE_COMPLEX, rank,
			0, MPI_COMM_WORLD, &stat);
}


/* Lehnsmann to Lehnsherr */
static int
distribute_one_payload(struct mfftw_plan *plan)
{
	fftw_complex *arr = reinterpret_numpy_to_fftw_arr(plan->out_arr);

	return transmit_payload(0, arr, plan->info->arrmeta.local_no);
}


/* Lehnsherr to Lehnsmaenner */
static int
distribute_all_payloads(struct mfftw_plan *plan)
{
	int i, ret = 0;
	struct mfftw_mpi_info *tmp_info;
	size_t tmp_i_start, tmp_ni;
	fftw_complex *in_arr;

	in_arr = reinterpret_numpy_to_fftw_arr(plan->in_arr);

	for (i = 1; i < plan->info->procmap.nr_of_procs; i++) {
		tmp_info = &plan->info->procmap.infos[i];
		tmp_i_start = tmp_info->arrmeta.local_i_start;
		tmp_ni = tmp_info->arrmeta.local_ni;

		ret = transmit_payload(i, &in_arr[tmp_i_start], tmp_ni);
		if (ret != 0)
			break;
	}

	return ret;
}


/* Lehnsmaenner receive from Lehnsherr */
static int
collect_one_payload(struct mfftw_plan *plan)
{
	fftw_complex *arr = reinterpret_numpy_to_fftw_arr(plan->in_arr);

	return receive_payload(0, arr, plan->info->arrmeta.local_ni);
}


/* Lehnsherr receives the results. */
static int
collect_all_payloads(struct mfftw_plan *plan)
{
	int i, ret = 0;
	struct mfftw_mpi_info *tmp_info;
	size_t tmp_o_start, tmp_no;
	fftw_complex *out_arr;

	out_arr = reinterpret_numpy_to_fftw_arr(plan->out_arr);

	for (i = 1; i < plan->info->procmap.nr_of_procs; i++) {
		tmp_info = &plan->info->procmap.infos[i];
		tmp_o_start = tmp_info->arrmeta.local_o_start;
		tmp_no = tmp_info->arrmeta.local_no;

		ret = receive_payload(i, &out_arr[tmp_o_start], tmp_no);
		if (ret != 0)
			break;
	}

	return ret;
}

#endif /* MFFTW_MPI */


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

#ifdef MFFTW_MPI
	if (mplan->info->rank == 0)
		distribute_all_payloads(mplan);
	else
		collect_one_payload(mplan);
#endif

	fftw_execute(mplan->plan);

#ifdef MFFTW_MPI
	if (mplan->info->rank == 0)
		collect_all_payloads(mplan);
	else
		distribute_one_payload(mplan);
#endif

	Py_INCREF(mplan->out_arr);
	return (PyObject *)(mplan->out_arr);
}


#ifdef MFFTW_MPI
static bool
initialize_threaded_mpi(PyObject *argv_list)
{
	int passed_argc = 0, provided = 0;

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
		PyErr_SetString(Mfftw_error, "Could not initialize threads.");
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
/*
 * Terminate all MPI-processes.
 * TODO: Verify if this is appropriate.
 */
	MPI_Finalize();
	exit(EXIT_SUCCESS);
#else
	fftw_cleanup();	
#endif /* MFFTW_MPI */

	Py_RETURN_NONE;
}


static PyMethodDef Minifftw_methods[] = {
#ifdef MFFTW_MPI
	{"init", init, METH_VARARGS, "prepare FFTW and MPI"},
	{"get_mpi_rank", get_mpi_rank, METH_VARARGS, "get MPI rank"},
	{"import_system_wisdom", import_system_wisdom_mpi, METH_VARARGS,
		"import the FFTW system-wisdom"},
	{"import_wisdom", import_wisdom_mpi, METH_VARARGS,
		"import wisdom and broadcast it over MPI"},
	{"export_wisdom", export_wisdom_mpi, METH_VARARGS,
		"gather wisdom over MPI and export it"},
	{"plan_dft_1d", plan_dft_1d_mpi, METH_VARARGS, "one dimensional FFT"},
#else
	{"init", init, METH_VARARGS, "prepare FFTW"},
	{"get_mpi_rank", get_pseudo_rank, METH_VARARGS, "get MPI pseudo rank"},
	{"import_system_wisdom", import_system_wisdom, METH_VARARGS,
		"import the FFTW system-wisdom"},
	{"import_wisdom", import_wisdom, METH_VARARGS,
		"import the FFTW wisdom from a filename/path"},
	{"export_wisdom", export_wisdom, METH_VARARGS,
		"export the FFTW wisdom to a filename/path"},
	{"plan_dft_1d", plan_dft_1d, METH_VARARGS, "one dimensional FFT"},
#endif /* MFFTW_MPI */
	{"finit", finit, METH_VARARGS, "finalize everything"},
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
	PyModule_AddIntMacro(m, FFTW_PRESERVE_INPUT);
	PyModule_AddIntMacro(m, FFTW_UNALIGNED);

	import_array();

	return m;
}
