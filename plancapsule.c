#include <Python.h>
#include <stdlib.h>
#include "util.h"
#include "minifftw.h"


PyCapsule
mfftw_create_capsule(struct mini_fftw_plan *plan)
{
	PyObject *capsule;
	capsule = PyCapsule_New((void*)plan, NULL, mfftw_destroy_capsule);
	return capsule;
}


void
mfftw_destroy_capsule(PyObject *capsule)
{
	struct mini_fftw_plan *plan = 
		(struct mini_fftw_plan *)PyCapsule_GetPointer(capsule);
	/* TODO error handling */

	mfftw_cleanup_plan(plan);
}


/*
 * Creates a new capsule structure which will later be passed into the Python
 * space as a general handler struct (in form of an opaque pointer)
 */
struct mini_fftw_plan *
mfftw_create_capsule_struct(fftw_plan plan, PyObject *orig_list, fftw_complex *arr)
{
	struct mini_fftw_plan *capsule = malloc(sizeof(struct mini_fftw_plan));
	if (!capsule) {
		perror("mfftw_create_plan_capsule");
		return NULL;
	}

	capsule->original_list = original_list;
	capsule->input_array = arr;
	capsule->plan = plan;

	return capsule;
}


void
mfftw_cleanup_plan(struct mini_fftw_plan *plan)
{
	free(plan->input_array);
	fftw_destroy(plan->plan);
	Py_DECREF(plan->original_list);
}
