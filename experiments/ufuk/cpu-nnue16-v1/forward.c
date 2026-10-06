#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#define EMB (12288*16)
#define TOTAL (EMB+17)
static double read_double(const char *p, int i) {
    double x; memcpy(&x, p+i*sizeof(double), sizeof(double)); return x;
}
static PyObject *forward(PyObject *self, PyObject *args) {
    PyObject *boards, *payload, *mover;
    int king;
    if (!PyArg_ParseTuple(args,"OiOO",&boards,&king,&mover,&payload)) return NULL;
    if (!PyBool_Check(mover) || king<0 || king>=64 || !PyBytes_Check(payload) ||
        PyBytes_Size(payload)!=TOTAL*sizeof(double)) {
        PyErr_SetString(PyExc_ValueError,"exact12 bitboards/king/mover/model bytes"); return NULL;
    }
    PyObject *seq=PySequence_Fast(boards,"12 bitboards required");
    if (!seq) return NULL;
    if (PySequence_Fast_GET_SIZE(seq)!=12) {
        Py_DECREF(seq); PyErr_SetString(PyExc_ValueError,"12 bitboards required"); return NULL;
    }
    uint64_t bb[12],used=0;
    for (int i=0;i<12;i++) {
        PyObject *v=PySequence_Fast_GET_ITEM(seq,i);
        if (!PyLong_CheckExact(v)) {
            Py_DECREF(seq); PyErr_SetString(PyExc_ValueError,"unsigned integer bitboard"); return NULL;
        }
        bb[i]=PyLong_AsUnsignedLongLong(v);
        if (PyErr_Occurred()) {Py_DECREF(seq);return NULL;}
        if (used&bb[i]) {
            Py_DECREF(seq);PyErr_SetString(PyExc_ValueError,"overlapping pieces");return NULL;
        }
        used|=bb[i];
    }
    Py_DECREF(seq);
    int own=mover==Py_True?5:11;
    if(!bb[5]||!bb[11]||(bb[5]&(bb[5]-1))||(bb[11]&(bb[11]-1))||
       !(bb[own]&((uint64_t)1<<king))){
        PyErr_SetString(PyExc_ValueError,"exact king bitboards/own king");return NULL;
    }
    int count=0; for(uint64_t v=used;v;v&=v-1)count++;
    if(count<1||count>32){PyErr_SetString(PyExc_ValueError,"piece count1..32");return NULL;}
    int white=mover==Py_True,ok=white?king:king^56;
    int bucket=(ok/8/2)*4+(ok%8/2);
    const char *data=PyBytes_AsString(payload);
    double hidden[16]={0};
    for(int r=0;r<12;r++) {
        int color=r<6?(white?0:6):(white?6:0);
        uint64_t pieces=bb[color+r%6];
        for(int square=0;square<64;square++) {
            int absolute=white?square:square^56;
            if(!(pieces&((uint64_t)1<<absolute)))continue;
            int index=bucket*768+r*64+square;
            for(int j=0;j<16;j++)hidden[j]+=read_double(data,index*16+j);
        }
    }
    double result=read_double(data,EMB+16);
    for(int j=0;j<16;j++)result+=tanh(hidden[j])*read_double(data,EMB+j);
    if(!isfinite(result)){PyErr_SetString(PyExc_ValueError,"finite residual required");return NULL;}
    return PyFloat_FromDouble(result);
}
static PyMethodDef methods[]={{"forward",forward,METH_VARARGS,"Sparse king-bucket residual only."},{NULL,NULL,0,NULL}};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_kingbucket16",NULL,-1,methods};
PyMODINIT_FUNC PyInit__kingbucket16(void){return PyModule_Create(&module);}
