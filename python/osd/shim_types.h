#include <Python.h>

#include <common/mutex.h>
#include <far/meshFactory.h>
#include <osd/vertex.h>
#include <osd/cpuComputeContext.h>
#include <osd/cpuVertexBuffer.h>

typedef OpenSubdiv::HbrMesh<OpenSubdiv::OsdVertex>     OsdHbrMesh;
typedef OpenSubdiv::HbrVertex<OpenSubdiv::OsdVertex>   OsdHbrVertex;
typedef OpenSubdiv::HbrFace<OpenSubdiv::OsdVertex>     OsdHbrFace;
typedef OpenSubdiv::HbrHalfedge<OpenSubdiv::OsdVertex> OsdHbrHalfedge;

// -----------------------------------------------------------------------------

struct OpaqueHbrMesh {
	PyObject_HEAD
    OsdHbrMesh *hmesh;
    std::vector<OsdHbrFace*>* faces;
};

static void
OpaqueHbrMesh_dealloc(OpaqueHbrMesh *self)
{
    self->ob_type->tp_free((PyObject*)self);
}

static PyMethodDef OpaqueHbrMesh_methods[] = {
	{NULL, NULL}
};

statichere PyTypeObject OpaqueHbrMesh_Type = {
    PyObject_HEAD_INIT(0)
    0,                      // ob_size
    "OpaqueHbrMesh",        // tp_name
    sizeof(OpaqueHbrMesh), // tp_basicsize
    0,                      // tp_itemsize
    (destructor) OpaqueHbrMesh_dealloc, // tp_dealloc
    0,          // tp_print
    0,          // tp_getattr
    0,          // tp_setattr
    0,          // tp_compare
    0,          // tp_repr
    0,          // tp_as_number
    0,          // tp_as_sequence
    0,          // tp_as_mapping
    0,          // tp_hash
    0,          // tp_call
    0,          // tp_str
    0,          // tp_getattro
    0,          // tp_setattro
    0,          // tp_as_buffer
    Py_TPFLAGS_DEFAULT,  // tp_flags
    "OpaqueHbrMesh Object",   //  tp_doc 
};

#define OpaqueHbrMesh_Check(v)	((v)->ob_type == &OpaqueHbrMesh_Type)

PyObject *
OpaqueHbrMesh_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    OpaqueHbrMesh *self = (OpaqueHbrMesh*) type->tp_alloc(type, 0);
    self->hmesh = NULL;
    return (PyObject*) self;
}

// -----------------------------------------------------------------------------

struct CSubd {
	PyObject_HEAD
    OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex>* farMesh;
    OpenSubdiv::OsdCpuComputeContext* computeContext;
    OpenSubdiv::OsdCpuVertexBuffer* vertexBuffer;
};

static void
CSubd_dealloc(CSubd *self)
{
    self->ob_type->tp_free((PyObject*)self);
}

static PyMethodDef CSubd_methods[] = {
	{NULL, NULL}
};

statichere PyTypeObject CSubd_Type = {
    PyObject_HEAD_INIT(0)
    0,                      // ob_size
    "CSubd",        // tp_name
    sizeof(CSubd), // tp_basicsize
    0,                      // tp_itemsize
    (destructor) CSubd_dealloc, // tp_dealloc
    0,          // tp_print
    0,          // tp_getattr
    0,          // tp_setattr
    0,          // tp_compare
    0,          // tp_repr
    0,          // tp_as_number
    0,          // tp_as_sequence
    0,          // tp_as_mapping
    0,          // tp_hash
    0,          // tp_call
    0,          // tp_str
    0,          // tp_getattro
    0,          // tp_setattro
    0,          // tp_as_buffer
    Py_TPFLAGS_DEFAULT,  // tp_flags
    "CSubd Object",   //  tp_doc 
};

#define CSubd_Check(v)	((v)->ob_type == &CSubd_Type)

PyObject *
CSubd_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    CSubd *self = (CSubd*) type->tp_alloc(type, 0);
    self->farMesh = 0;
    self->computeContext = 0;
    self->vertexBuffer = 0;
    return (PyObject*) self;
}
