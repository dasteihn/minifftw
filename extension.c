#define PY_SSIZE_T_CLEAN
#include <Python.h>
/* #include <fftw3-mpi.h> */
#include <stdio.h>
#include <stdbool.h>
#include "util.h"

static PyObject *Fftw_error = NULL;


static PyObject* parse_complex(PyObject *self, PyObject *args)
{
	Py_ssize_t list_len = 0;
	PyObject *list = NULL;
	Py_complex *array = NULL;

	puts("alive 0");
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
	{NULL, NULL, 0, NULL},
};


static struct PyModuleDef fftwmodule = {
	PyModuleDef_HEAD_INIT,
	"comparse", /* test: parse complex numbers */
	NULL,
	-1, /* awkward interpreter state foo */
	Minifftw_methods,
};

	
PyMODINIT_FUNC PyInit_minifftw(void)
{
	PyObject *m;

	m = PyModule_Create(&fftwmodule);
	if (!m)
		return NULL;

	Fftw_error = PyErr_NewException("spam.error", NULL, NULL);
	Py_XINCREF(Fftw_error);
	if (PyModule_AddObject(m, "error", Fftw_error) < 0) {
		Py_XDECREF(Fftw_error);
		Py_CLEAR(Fftw_error);
		Py_DECREF(m);
		return NULL;
	}

	return m;
}
