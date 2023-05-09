#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

static PyObject* nearest_neighbors(PyObject* self, PyObject* args) {
    PyArrayObject* r0_obj;
    PyArrayObject* r_obj;
    int n;

    if (!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &r0_obj, &PyArray_Type, &r_obj, &n)) {
        return NULL;
    }

    int r_len = PyArray_DIM(r_obj, 0);

    double distances[r_len];
    int indices[r_len];

    for (int i = 0; i < r_len; i++) {
        double dx = *(double*)PyArray_GETPTR1(r0_obj, 0) - *(double*)PyArray_GETPTR2(r_obj, i, 0);
        double dy = *(double*)PyArray_GETPTR1(r0_obj, 1) - *(double*)PyArray_GETPTR2(r_obj, i, 1);
        distances[i] = sqrt(dx * dx + dy * dy);
        indices[i] = i;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < r_len - i - 1; j++) {
            if (distances[j] > distances[j + 1]) {
                double temp = distances[j];
                distances[j] = distances[j + 1];
                distances[j + 1] = temp;

                int temp_idx = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = temp_idx;
            }
        }
    }

    npy_intp dims[1] = {n};
    PyObject* result = PyArray_SimpleNew(1, dims, NPY_INT);

    for (int i = 0; i < n; i++) {
        int* buffer = (int*)PyArray_GETPTR1((PyArrayObject*)result, i);
        *buffer = indices[i];
    }

    return result;
}


static PyMethodDef NearestNeighborsMethods[] = {
    {"nearest_neighbors", nearest_neighbors, METH_VARARGS, "Find nearest neighbors."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef nearestneighborsmodule = {
    PyModuleDef_HEAD_INIT,
    "nearest_neighbors",
    NULL,
    -1,
    NearestNeighborsMethods
};

PyMODINIT_FUNC PyInit_nearest_neighbors(void) {
    import_array();
    return PyModule_Create(&nearestneighborsmodule);
}

