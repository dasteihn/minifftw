#define PY_SSIZE_T_CLEAN
#include <Python.h>
/* #include <fftw3-mpi.h> */
#include <stdio.h>

static PyObject *Fftw_error = NULL;


static PyObject* parse_complex(PyObject *self, PyObject *args)
{
	PyObject *data = NULL;
	puts("alive 0");
	int ret = PyArg_ParseTuple(args, "O!", &PyList_Type, &data);
	puts("alive 1");
	if (!data) return NULL;
	puts("alive 2");
	if (ret == 0) return NULL;

	if (PyList_Check(data) == 0)
		puts("not a list :(");
	else
		puts("a list! :)");

	puts("alive 3");
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
