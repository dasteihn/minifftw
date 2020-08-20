#ifndef __minifftw_
#define __minifftw_

#include <Python.h>
#include <fftw3.h>

bool is_complex_list(PyObject *);
Py_complex* complex_list_to_c_array(PyObject *);

struct mini_fftw_plan {
	PyObject *original_list;
	fftw_complex *input_array;
	fftw_plan plan;
};


#endif /* __minifftw_ */
