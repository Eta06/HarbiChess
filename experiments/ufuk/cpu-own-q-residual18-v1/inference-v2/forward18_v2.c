#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <string.h>

/* Host-native doubles, verified ABI and layout by the sealed Python wrapper.
 * No fast-math, no floating-point contraction. No board/rule/search work here. */
static PyObject *forward(PyObject *self, PyObject *args) {
    PyObject *raw, *payload; double prior;
    if (!PyArg_ParseTuple(args, "OOd", &raw, &payload, &prior)) return NULL;
    if (!PyBytes_Check(payload) || PyBytes_Size(payload) != 660 * sizeof(double)) {
        PyErr_SetString(PyExc_ValueError, "exact sealed 660-double payload required");
        return NULL;
    }
    PyObject *seq = PySequence_Fast(raw, "18 raw features required");
    if (!seq) return NULL;
    if (PySequence_Fast_GET_SIZE(seq) != 18) {
        Py_DECREF(seq); PyErr_SetString(PyExc_ValueError, "18 raw features required");
        return NULL;
    }
    double data[660], x[18], hidden[32];
    memcpy(data, PyBytes_AsString(payload), sizeof(data));
    if (!isfinite(prior)) { Py_DECREF(seq); PyErr_SetString(PyExc_ValueError, "finite exact Python prior required"); return NULL; }
    for (int i = 0; i < 18; i++) {
        x[i] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(seq, i));
        if (PyErr_Occurred()) { Py_DECREF(seq); return NULL; }
        if (!isfinite(x[i])) {
            Py_DECREF(seq); PyErr_SetString(PyExc_ValueError, "nonfinite raw feature");
            return NULL;
        }
    }
    Py_DECREF(seq);
    for (int j = 0; j < 32; j++) {
        double z = data[576 + j];
        for (int i = 0; i < 18; i++) z += x[i] * data[i * 32 + j];
        hidden[j] = tanh(z);
    }
    double residual = data[640];
    for (int j = 0; j < 32; j++) residual += hidden[j] * data[608 + j];
    double result = tanh(residual == 0.0 ? prior : prior + residual);
    if (!isfinite(result) || result < -1.0 || result > 1.0) {
        PyErr_SetString(PyExc_ValueError, "finite mover value required"); return NULL;
    }
    return PyFloat_FromDouble(result);
}
static PyMethodDef methods[] = {{"forward", forward, METH_VARARGS, "Fixed 18x32 scalar forward."}, {NULL,NULL,0,NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_ownq_forward18_v2", NULL, -1, methods};
PyMODINIT_FUNC PyInit__ownq_forward18_v2(void) { return PyModule_Create(&module); }
