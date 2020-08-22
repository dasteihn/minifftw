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

#include <Python.h>
#include <stdlib.h>
#include "minifftw.h"

/* Capsule creation and destruction */

static void
mfftw_cleanup_plan(struct mini_fftw_plan *plan)
{
	free(plan->input_array);
	free(plan->output_array);
	fftw_destroy_plan(plan->plan);
	Py_DECREF(plan->original_list);
}


static void
mfftw_destroy_capsule(PyObject *capsule)
{
	struct mini_fftw_plan *plan = 
		(struct mini_fftw_plan *)PyCapsule_GetPointer(capsule, NULL);
	/* TODO error handling */

	mfftw_cleanup_plan(plan);
}


static inline PyObject *
mfftw_create_capsule(struct mini_fftw_plan *mplan)
{
	return PyCapsule_New((void*)mplan, NULL, mfftw_destroy_capsule);
}


/*
 * Creates a new capsule structure which will later be passed into the Python
 * space as a general handler struct (in form of an opaque pointer).
 * The capsule-struct will also contain the python-list, so we call INCREF.
 */
static struct mini_fftw_plan *
mfftw_create_capsule_struct(fftw_plan plan, PyObject *orig_list,
		fftw_complex *in_arr, fftw_complex *out_arr)
{
	struct mini_fftw_plan *capsule = calloc(1, sizeof(struct mini_fftw_plan));
	if (!capsule) {
		return NULL;
	}

	capsule->data_len = PyList_Size(orig_list);
	capsule->original_list = orig_list;
	Py_INCREF(orig_list);
	capsule->input_array = in_arr;
	capsule->output_array = out_arr;
	capsule->plan = plan;

	return capsule;
}


/*
 * Creates a python capsule containing everything the Python-world needs to
 * interact with the FFTW.
 */
PyObject *
mfftw_encapsulate_plan(fftw_plan plan, PyObject *orig_list,
		fftw_complex *in_arr, fftw_complex *out_arr)
{
	struct mini_fftw_plan *mplan = NULL;
	mplan = mfftw_create_capsule_struct(plan, orig_list, in_arr, out_arr);
	if (!mplan)
		return PyErr_NoMemory();

	return mfftw_create_capsule(mplan);

}


/* 
 * ========================= Capsule evaluation ===============================
 */

mini_fftw_plan *
mfftw_unwrap_capsule(PyObject *mplan)
{
	if (PyCapsule_CheckExact(plancapsule) == 0) {
		PyErr_SetString(PyExc_TypeError, "Expected a capsule.");
		return NULL;
	}
	struct mini_fftw_plan *plan =
		(struct mini_fftw_plan *)PyCapsule_GetPointer(mplan);
	// TODO error handling?
	return plan;
}


/* Actually copies the contents of the python list into the C array */
int
mfftw_prepare_for_execution(mini_fftw_plan *mplan)
{
	int ret = -1;
	ret = fill_fftw_array(mplan->orig_list, mplan->input_array, mplan->list_len);
	if (ret != 0) {
		PyErr_SetString(PyExc_StopIteration, "Stored data list not iterable.");
	}

	return ret;
}


int
mfftw_mplan_prepare_for_output(mini_fftw_plan *mplan)
{
	return fftw_arr_to_list(mplan->orig_list, mplan->output_array, mplan->list_len);
}
