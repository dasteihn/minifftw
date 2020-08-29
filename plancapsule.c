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
#define NPY_NO_DEPRECATED_API  NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include "minifftw.h"

/*
 * ====================== Capsule creation and destruction ===================
 */

static void
mfftw_cleanup_plan(struct mfftw_plan *plan)
{
	/* FIXME: use fftw_free */
	free(plan->input_arr);
	free(plan->output_arr);
	fftw_destroy_plan(plan->plan);
	Py_DECREF(plan->orig_arr);
}


static void
mfftw_destroy_capsule(PyObject *capsule)
{
	struct mfftw_plan *plan = 
		(struct mfftw_plan *)PyCapsule_GetPointer(capsule, NULL);
	/* TODO error handling */

	mfftw_cleanup_plan(plan);
}


static inline PyObject *
mfftw_create_capsule(struct mfftw_plan *mplan)
{
	return PyCapsule_New((void*)mplan, NULL, mfftw_destroy_capsule);
}


/*
 * Creates a new capsule structure which will later be passed into the Python
 * space as a general handler struct (in form of an opaque pointer).
 * The capsule-struct will also contain the python-list, so we call INCREF.
 */
static struct mfftw_plan *
mfftw_create_capsule_struct(fftw_plan plan, PyObject *original_arr,
		fftw_complex *in_arr, fftw_complex *out_arr)
{
	struct mfftw_plan *capsule = calloc(1, sizeof(struct mfftw_plan));
	if (!capsule) {
		return NULL;
	}

	capsule->data_len = PyArray_SIZE(original_arr);
	capsule->orig_arr = original_arr;
	Py_INCREF(original_arr);
	capsule->input_arr = in_arr;
	capsule->output_arr = out_arr;
	capsule->plan = plan;

	return capsule;
}


/*
 * Creates a python capsule containing everything the Python-world needs to
 * interact with the FFTW.
 */
PyObject *
mfftw_encapsulate_plan(fftw_plan plan, PyObject *orig_arr,
		fftw_complex *in_arr, fftw_complex *out_arr)
{
	struct mfftw_plan *mplan = NULL;
	mplan = mfftw_create_capsule_struct(plan, orig_arr, in_arr, out_arr);
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
	struct mfftw_plan *plan =
		(struct mfftw_plan *)PyCapsule_GetPointer(mplan, NULL);
	// TODO error handling?
	return plan;
}


/* Actually copies the contents of the python list into the C array */
int
mfftw_prepare_for_execution(struct mfftw_plan *mplan)
{
	mfftw_data_from_npy_to_fftw(mplan->orig_arr, mplan->input_arr,
			mplan->data_len);
	return 0;
}


int
mfftw_prepare_for_output(struct mfftw_plan *mplan)
{
	mfftw_data_from_fftw_to_npy(mplan->orig_arr, mplan->output_arr,
		mplan->data_len);
	return 0;
}
