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

#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL mfftw_ARRAY_API
#include <numpy/arrayobject.h>

#include <stdlib.h>
#include "minifftw.h"

/*
 * ====================== Capsule creation and destruction ===================
 */


static void
mfftw_destroy_capsule(PyObject *capsule)
{
	struct mfftw_plan *plan = 
		(struct mfftw_plan *)PyCapsule_GetPointer(capsule, NULL);

	fftw_destroy_plan(plan->plan);
	Py_DECREF(plan->in_arr);
	Py_DECREF(plan->out_arr);

	/* When MPI is not used, this is NULL */
	if (plan->info) {
		free(plan->info->arrmeta);
		free(plan->info);
	}

	free(plan);
}


static inline PyObject *
mfftw_create_capsule(struct mfftw_plan *mplan)
{
	return PyCapsule_New((void*)mplan, NULL, mfftw_destroy_capsule);
}


/*
 * Creates a new capsule structure which will later be passed into the Python
 * space as a general handler struct (in form of an opaque pointer).
 * The capsule-struct will also contain the numpy-arrays, so we call INCREF.
 */
static struct mfftw_plan *
mfftw_create_capsule_struct(fftw_plan plan,
		PyArrayObject *in_arr, PyArrayObject *out_arr,
		struct mfftw_mpi_info *info)
{
	struct mfftw_plan *capsule = calloc(1, sizeof(struct mfftw_plan));
	if (!capsule) {
		/* The error is handled by the caller */
		return NULL;
	}

	capsule->data_len = PyArray_SIZE(in_arr);
	Py_INCREF(in_arr);
	/* The arrays might be identical, but we don't care, we will also
	 * DECREF them two times. */
	Py_INCREF(out_arr);
	capsule->in_arr = in_arr;
	capsule->out_arr = out_arr;
	capsule->plan = plan;
	capsule->info = info;

	return capsule;
}


/*
 * Creates a python capsule containing everything the Python-world needs to
 * interact with the FFTW.
 */
PyObject *
mfftw_encapsulate_plan(fftw_plan plan,
		PyArrayObject *in_arr, PyArrayObject *out_arr,
		struct mfftw_mpi_info *info)
{
	struct mfftw_plan *mplan = NULL;
	mplan = mfftw_create_capsule_struct(plan, in_arr, out_arr, info);
	if (!mplan)
		return PyErr_NoMemory();

	return mfftw_create_capsule(mplan);
}


/* 
 * ========================= Capsule evaluation ===============================
 */

struct mfftw_plan *
mfftw_unwrap_capsule(PyObject *mplan)
{
	if (PyCapsule_CheckExact(mplan) == 0) {
		PyErr_SetString(PyExc_TypeError, "Expected a capsule.");
		return NULL;
	}

	return (struct mfftw_plan *)PyCapsule_GetPointer(mplan, NULL);
}
