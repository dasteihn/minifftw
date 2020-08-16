#ifndef __minifftw_util
#define __minifftw_util

#include <Python.h>
#include <stdbool.h>

bool is_complex_list(PyObject *);
Py_complex* complex_list_to_c_array(PyObject *);

#endif /* __minifftw_util */
