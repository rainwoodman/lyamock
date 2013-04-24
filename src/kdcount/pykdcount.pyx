#cython: embedsignature=True
#cython: cdivision=True
cimport numpy
import numpy
from cpython.mem cimport PyMem_Malloc, PyMem_Free, PyMem_Realloc
from libc.stdint cimport intptr_t as npy_intp
numpy.import_array()

cdef extern from "kdcount.h":
    ctypedef double (*kd_castfunc)(void * p)
    ctypedef int (*kd_enum_callback)(void * data, double r, npy_intp i, npy_intp j)
    struct cKDType "KDType":
        char * buffer
        npy_intp size
        int Nd
        npy_intp strides[2]
        int thresh
        npy_intp * ind
        double * boxsize
        double p
        npy_intp elsize
        double (* cast)(void * p1)
        void * (*malloc)(npy_intp size)
        void * (*free)(npy_intp size, void * ptr)
        void * userdata

    struct cKDNode "KDNode":
        cKDType * type
        cKDNode * link[2]
        npy_intp start
        npy_intp size
        int dim
        double split

    cKDNode * kd_build(cKDType * type)
    double * kd_node_max(cKDNode * node)
    double * kd_node_min(cKDNode * node)
    void kd_free(cKDNode * node)
    void kd_free0(cKDType * type, npy_intp size, void * ptr)
    cKDNode ** kd_split(cKDNode * node, npy_intp thresh, npy_intp * length)
    int kd_enum(cKDNode * node[2], double maxr,
            kd_enum_callback callback, void * data) except -1

cdef class KDNode:
    cdef cKDNode * ref
    cdef readonly KDType type
    def __init__(self, type):
        self.type = type

    cdef void bind(self, cKDNode * ref) nogil:
        self.ref = ref

    property less:
        def __get__(self):
            cdef KDNode rt = KDNode(self.type)
            rt.bind(self.ref.link[0])
            return rt

    property greater:
        def __get__(self):
            cdef KDNode rt = KDNode(self.type)
            rt.bind(self.ref.link[1])
            return rt

    property start:
        def __get__(self):
            return self.ref.start
    
    property size:
        def __get__(self):
            return self.ref.size

    property dim:
        def __get__(self):
            return self.ref.dim

    property split:
        def __get__(self):
            return self.ref.split

    property max:
        def __get__(self):
            cdef double * max = kd_node_max(self.ref)
            return [max[d] for d in range(self.ref.type.Nd)]

    property min:
        def __get__(self):
            cdef double * min = kd_node_min(self.ref)
            return [min[d] for d in range(self.ref.type.Nd)]

    def __repr__(self):
        return str((self.dim, self.split, self.size))

    def subtrees(self, thresh):
        cdef cKDNode ** list
        cdef npy_intp len
        list = kd_split(self.ref, thresh, &len)
        cdef npy_intp i
        ret = [KDNode(self.type) for i in range(len)]
        for i in range(len):
            (<KDNode>(ret[i])).bind(list[i])
        kd_free0(self.type.ref, len * sizeof(cKDNode*), list)
        return ret

    def enum(self, KDNode other, process, rmax, bunch=10000, **kwargs):
        cdef numpy.ndarray r = numpy.empty(bunch, 'f8')
        cdef numpy.ndarray i = numpy.empty(bunch, 'intp')
        cdef numpy.ndarray j = numpy.empty(bunch, 'intp')

        cdef cKDNode * node[2]
        cdef CBData cbdata
        def func():
            process(r[:cbdata.length], i[:cbdata.length], j[:cbdata.length],
                    **kwargs)
        node[0] = self.ref
        node[1] = other.ref
        cbdata.notify = <void*>func
        cbdata.r = <double*>r.data
        cbdata.i = <npy_intp*>i.data
        cbdata.j = <npy_intp*>j.data
        cbdata.size = bunch
        cbdata.length = 0
        kd_enum(node, rmax, <kd_enum_callback>callback, &cbdata)

cdef double dcast(double * p1) nogil:
    return p1[0]
cdef double fcast(float * p1) nogil:
    return p1[0]

cdef struct CBData:
    double * r
    npy_intp * i
    npy_intp * j
    npy_intp size
    npy_intp length
    void * notify

cdef int callback(CBData * data, double r, npy_intp i, npy_intp j) except -1:
    if data.length == data.size:
        (<object>(data.notify)).__call__()
        data.length = 0
    data.r[data.length] = r
    data.i[data.length] = i
    data.j[data.length] = j
    data.length = data.length + 1
    return 0

cdef class KDType:
    cdef cKDType * ref
    cdef cKDNode * tree
    cdef readonly numpy.ndarray input
    cdef readonly numpy.ndarray ind
    cdef readonly numpy.ndarray boxsize
    property root:
        def __get__(self):
            cdef KDNode rt = KDNode(self)
            rt.bind(self.tree)
            return rt

    def __init__(self, numpy.ndarray input, boxsize=None):
        self.input = input
        self.ref = <cKDType*>PyMem_Malloc(sizeof(cKDType))
        self.ref.buffer = input.data
        self.ref.size = input.shape[0]
        self.ref.Nd = input.shape[1]
        self.ref.strides[0] = input.strides[0]
        self.ref.strides[1] = input.strides[1]
        self.ref.thresh = 10
        self.ind = numpy.empty(self.ref.size, dtype='intp')
        self.ref.ind = <npy_intp*> self.ind.data
        if boxsize != None:
            self.boxsize = numpy.empty(self.ref.Nd, dtype='f8')
            self.boxsize[:] = boxsize
            self.ref.boxsize = <double*>self.boxsize.data
            print self.ref.boxsize[0], self.ref.boxsize[1]
        else:
            self.boxsize = None
            self.ref.boxsize = NULL
        self.ref.elsize = input.dtype.itemsize
        if input.dtype.char == 'f':
            self.ref.cast = <kd_castfunc>fcast
        if input.dtype.char == 'd':
            self.ref.cast = <kd_castfunc>dcast
        self.ref.malloc = NULL
        self.ref.free = NULL
        self.tree = kd_build(self.ref)


    def __dealloc__(self):
        if self.tree:
            kd_free(self.tree)
        PyMem_Free(self.ref)

def build(numpy.ndarray data, boxsize=None):
    type = KDType(data, boxsize)
    return type.root

