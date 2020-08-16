#include <Python.h>
#include <stdbool.h>
#include <stdio.h>


bool
is_complex_list(PyObject *o)
{
	PyObject *first_item = PyList_GetItem(o, 0);
	return (bool)PyComplex_Check(first_item);
}


static void
fill_array(PyObject *list, Py_complex *array, Py_ssize_t total_len)
{
	puts("entered fill array");
	size_t i;
	PyObject *iterator = PyObject_GetIter(list);
	if (!iterator || PyIter_Check(iterator) == 0) {
		puts("is not iterable");
		return;
	}
	puts("is iterable");
	PyObject *iter_obj = NULL;

	for (i = 0; (iter_obj = PyIter_Next(iterator)) && i <= total_len; i++) {
		printf("looping %i\n", i);
		array[i] = PyComplex_AsCComplex(iter_obj);
		Py_DECREF(iter_obj);
	}

	Py_DECREF(iterator);
}


Py_complex *
complex_list_to_c_array(PyObject *list)
{
	Py_ssize_t list_len = PyList_Size(list);
	Py_complex *array = calloc(list_len, sizeof(Py_complex));
	if (!array) {
		perror("complex_list_to_c_array");
		return NULL;
	}

	fill_array(list, array, list_len);

	return array;
}
