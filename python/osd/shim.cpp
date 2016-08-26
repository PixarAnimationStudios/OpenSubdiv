
// These functions do not need to be incredibly user friendly (in
// terms of error checking) because the client is the Python portion
// of the osd wrapper, rather than the end user.  The C++ side of the
// shim is quite verbose so let's try to keep most of the sanity
// checking on the Python side of things.
//
// Useful references:
//   http://docs.scipy.org/doc/numpy/user/c-info.how-to-extend.html
//   http://dsnra.jpl.nasa.gov/software/Python/numpydoc/numpy-13.html

#include <Python.h>
#include <numpy/arrayobject.h>
#include <osd/cpuComputeController.h>
#include "shim_types.h"
#include "shim_adapters.cpp"

// These flags are useful for disabling the core OpenSubdiv library
// to help determine where problems (such as leaks) occur.
static const bool HBR_STUBBED = false;
static const bool FAR_STUBBED = false;

OpenSubdiv::OsdCpuComputeController * g_osdComputeController = 0;

// - args is a list with 1 element, which is a Topology object
// - returns an opaque handle to a HbrMesh (shim.OpaqueHbrMesh)
static PyObject *
HbrNew(PyObject *self, PyObject *args)
{
    Py_ssize_t n = PyTuple_Size(args);
    if (n != 1) {
        PyErr_SetString(PyExc_TypeError, "hbr_new requires a single argument.");
        return NULL;
    }
    
    PyObject* topology = PyTuple_GetItem(args, 0);
    PyObject* boundaryInterp =
        PyObject_GetAttrString(topology, "boundaryInterpolation");
    PyObject* indices = PyObject_GetAttrString(topology, "indices");
    PyObject* valences = PyObject_GetAttrString(topology, "valences");
    if (!boundaryInterp || !indices || !valences) {
        PyErr_SetString(PyExc_TypeError, "hbr_new requires a Topology object.");
        return NULL;
    }

    static OpenSubdiv::HbrCatmarkSubdivision<OpenSubdiv::OsdVertex>  _catmark;
    OsdHbrMesh* hmesh = 0;

    if (!HBR_STUBBED) {
        hmesh = new OsdHbrMesh(&_catmark);

        // TODO - allow input indices to be unsigned and/or 16 bits each

        long vertexCount = 1 + PyInt_AsLong(
            PyObject_GetAttrString(topology, "_maxIndex"));

        OpenSubdiv::OsdVertex vert;
        for (long i = 0; i < vertexCount; ++i) {
            OpenSubdiv::HbrVertex<OpenSubdiv::OsdVertex>* pVert =
                hmesh->NewVertex((int) i, vert);
            if (!pVert) {
                printf("Error: Unable to create vertex %ld\n", i);
            }
        }

        int* pIndices = (int*) PyArray_DATA(indices);
        unsigned char* pValence = (unsigned char*) PyArray_DATA(valences);
        Py_ssize_t valenceCount = PySequence_Length(valences);
        while (valenceCount--) {
            int vertsPerFace = *pValence;
            OpenSubdiv::HbrFace<OpenSubdiv::OsdVertex>* pFace =
                hmesh->NewFace(vertsPerFace, pIndices, 0);
            if (!pFace) {
                printf("Error: Unable to create face (valence = %d)\n",
                       vertsPerFace);
            }
            pIndices += vertsPerFace;
            ++pValence;
        }

        OsdHbrMesh::InterpolateBoundaryMethod bm =
            (OsdHbrMesh::InterpolateBoundaryMethod) PyInt_AsLong(boundaryInterp);
        hmesh->SetInterpolateBoundaryMethod(bm);
    }

    Py_DECREF(boundaryInterp);
    Py_DECREF(indices);
    Py_DECREF(valences);

    OpaqueHbrMesh *retval = PyObject_New(OpaqueHbrMesh,
                                         &OpaqueHbrMesh_Type);
    retval->hmesh = hmesh;
    retval->faces = new std::vector<OsdHbrFace*>();
    Py_INCREF(retval);
    return (PyObject *) retval;
}

static PyObject *
HbrFinish(PyObject *self, PyObject *args)
{
    OpaqueHbrMesh* hmesh = (OpaqueHbrMesh*) PyTuple_GetItem(args, 0);
    hmesh->hmesh->Finish();
    Py_INCREF(Py_None);
    return Py_None;
}

// - args is a list with 1 element, which is an opaque HbrMesh handle
// - returns None
static PyObject *
HbrDelete(PyObject *self, PyObject *args)
{
    OpaqueHbrMesh* hmesh = (OpaqueHbrMesh*) PyTuple_GetItem(args, 0);
    delete hmesh->hmesh;
    delete hmesh->faces;
    OpaqueHbrMesh_dealloc(hmesh);
    Py_INCREF(Py_None);
    return Py_None;
}

// - args is a list with 2 objects (Subdivider, Topology)
// - returns an opaque handle to a FarMesh (shim.CSubd)
static PyObject *
CSubdNew(PyObject *self, PyObject *args)
{
    Py_ssize_t n = PyTuple_Size(args);
    if (n != 2) {
        PyErr_SetString(PyExc_TypeError, "csubd_new requires two arguments.");
        return NULL;
    }

    PyObject* subdivider = PyTuple_GetItem(args, 0);
    PyObject* topology = PyTuple_GetItem(args, 1);

    OsdHbrMesh* hmesh;
    {
        OpaqueHbrMesh* handle;
        handle = (OpaqueHbrMesh*) PyObject_GetAttrString(topology, "_hbr_mesh");
        if (!handle || (!handle->hmesh && !HBR_STUBBED && !FAR_STUBBED)) {
            PyErr_SetString(PyExc_TypeError,
                            "csubd_new requires a finalized topology object.");
            return NULL;
        }
        hmesh = handle->hmesh;
        Py_DECREF(handle);
    }

    int level;
    {
        PyObject* levelObject = PyObject_GetAttrString(subdivider, "level");
        level = (int) PyInt_AsLong(levelObject);
        Py_DECREF(levelObject);
    }

    int numFloatsPerVertex = 0;
    {
        PyObject* vertexLayout = PyObject_GetAttrString(
            subdivider, "vertexLayout");

        PyObject* seq = PySequence_Fast(vertexLayout, "expected a sequence");
        Py_ssize_t len = PySequence_Size(seq);
        for (Py_ssize_t i = 0; i < len; i++) {
            PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
            item = PyTuple_GetItem(item, 1);
            if (!PyType_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "Bad vertex layout.");
                return NULL;
            }
            std::string typeName = ((PyTypeObject*) item)->tp_name;
            if (typeName == "numpy.float32") {
                ++numFloatsPerVertex;
            } else {
                PyErr_SetString(
                    PyExc_TypeError,
                    "Types other than numpy.float32 are not yet supported.");
                return NULL;
            }
        }
        
        Py_DECREF(seq);
        Py_DECREF(vertexLayout);
    }

    OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex>* farMesh = 0;
    OpenSubdiv::OsdCpuComputeContext* computeContext = 0;
    OpenSubdiv::OsdCpuVertexBuffer* vertexBuffer = 0;
    if (!FAR_STUBBED) {
        OpenSubdiv::FarMeshFactory<OpenSubdiv::OsdVertex> meshFactory(
            hmesh,
            level);
        farMesh = meshFactory.Create();
        computeContext = OpenSubdiv::OsdCpuComputeContext::Create(farMesh);
        vertexBuffer = OpenSubdiv::OsdCpuVertexBuffer::Create(
            numFloatsPerVertex, farMesh->GetNumVertices());
    }

    CSubd *retval = PyObject_New(CSubd, &CSubd_Type);
    retval->farMesh = farMesh;
    retval->computeContext = computeContext;
    retval->vertexBuffer = vertexBuffer;
    Py_INCREF(retval);
    return (PyObject *) retval;
}

// - args is a list with 1 element
// - returns None
static PyObject *
CSubdDelete(PyObject *self, PyObject *args)
{
    CSubd* csubd = (CSubd*) PyTuple_GetItem(args, 0);
    delete csubd->computeContext;
    delete csubd->farMesh;
    delete csubd->vertexBuffer;
    CSubd_dealloc(csubd);
    Py_INCREF(Py_None);
    return Py_None;
}

// - args is a list with 2 objects (CSubd and numpy array)
// - calls UpdateData on the vertexBuffer.
static PyObject *
CSubdUpdate(PyObject *self, PyObject *args)
{
    PyObject* csubdObject = PyTuple_GetItem(args, 0);
    PyObject* arrayObject = PyTuple_GetItem(args, 1);

    if (!CSubd_Check(csubdObject) || !PyArray_Check(arrayObject)) {
        PyErr_SetString(PyExc_TypeError, "csubd_update");
        return NULL;
    }

    CSubd* csubd = (CSubd*) csubdObject;
    PyArrayObject* coarseVerts = (PyArrayObject*) arrayObject;

    float* pFloats = (float*) PyArray_DATA(coarseVerts);
    int numFloats = PyArray_NBYTES(coarseVerts) / sizeof(float);

    csubd->vertexBuffer->UpdateData(pFloats, numFloats);

    Py_INCREF(Py_None);
    return Py_None;
}

// - args is a list with 1 objects (CSubd)
// - Calls Refine on the compute controller, passing it the compute
//   context and vertexBuffer.
static PyObject *
CSubdRefine(PyObject *self, PyObject *args)
{
    PyObject* csubdObject = PyTuple_GetItem(args, 0);

    if (!CSubd_Check(csubdObject)) {
        PyErr_SetString(PyExc_TypeError, "csubd_refine");
        return NULL;
    }

    CSubd* csubd = (CSubd*) csubdObject;
    g_osdComputeController->Refine(
        csubd->computeContext,
        csubd->vertexBuffer);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *
CSubdGetVerts(PyObject *self, PyObject *args)
{
    CSubd* csubd = (CSubd*) PyTuple_GetItem(args, 0);

    float* pFloats = csubd->vertexBuffer->BindCpuBuffer();

    npy_intp numFloats = (npy_intp)
        (csubd->vertexBuffer->GetNumElements() * 
         csubd->vertexBuffer->GetNumVertices());

    PyObject* retval = PyArray_SimpleNewFromData(
        1, &numFloats, PyArray_FLOAT, (void*) pFloats);

    Py_INCREF(retval);
    return retval;
}

static PyObject *
CSubdGetQuads(PyObject *self, PyObject *args)
{
    CSubd* csubd = (CSubd*) PyTuple_GetItem(args, 0);
    PyObject* dtype = PyTuple_GetItem(args, 1);

    std::string typeName = ((PyTypeObject*) dtype)->tp_name;
    if (typeName != "numpy.uint32") {
        PyErr_SetString(
            PyExc_TypeError,
            "Index types other than numpy.uint32 are not yet supported.");
        return NULL;
    }
    
    OpenSubdiv::FarPatchTables const * patchTables =
        csubd->farMesh->GetPatchTables();

    if (patchTables) {
        PyErr_SetString(PyExc_TypeError, "feature adaptive not supported");
        return NULL;
    }

    const OpenSubdiv::FarSubdivisionTables<OpenSubdiv::OsdVertex> *tables =
        csubd->farMesh->GetSubdivisionTables();

    bool loop = dynamic_cast<const OpenSubdiv::FarLoopSubdivisionTables<
        OpenSubdiv::OsdVertex>*>(tables);

    if (loop) {
        PyErr_SetString(PyExc_TypeError, "loop subdivision not supported");
        return NULL;
    }

    int level = tables->GetMaxLevel();
    const std::vector<int> &indices = csubd->farMesh->GetFaceVertices(level-1);
    npy_intp length = (npy_intp) indices.size();

    // this does NOT create a copy
    PyObject* retval = PyArray_SimpleNewFromData(
        1, &length, PyArray_UINT, (void*) &indices[0]);

    Py_INCREF(retval);
    return retval;
}

static PyMethodDef osdFreeFunctions[] = {
    {"hbr_new", HbrNew, METH_VARARGS},
    {"hbr_delete", HbrDelete, METH_VARARGS},
    {"hbr_finish", HbrFinish, METH_VARARGS},
    {"csubd_new", CSubdNew, METH_VARARGS},
    {"csubd_delete", CSubdDelete, METH_VARARGS},
    {"csubd_update", CSubdUpdate, METH_VARARGS},
    {"csubd_refine", CSubdRefine, METH_VARARGS},
    {"csubd_getquads", CSubdGetQuads, METH_VARARGS},
    {"csubd_getverts", CSubdGetVerts, METH_VARARGS},

    {"hbr_update_faces", HbrUpdateFaces, METH_VARARGS},
    {"hbr_get_vertex_sharpness", HbrGetVertexSharpness, METH_VARARGS},
    {"hbr_set_vertex_sharpness", HbrSetVertexSharpness, METH_VARARGS},
    {"hbr_get_num_faces", HbrGetNumFaces, METH_VARARGS},
    {"hbr_get_face_hole", HbrGetFaceHole, METH_VARARGS},
    {"hbr_set_face_hole", HbrSetFaceHole, METH_VARARGS},
    {"hbr_get_num_edges", HbrGetNumEdges, METH_VARARGS},
    {"hbr_get_edge_sharpness", HbrGetEdgeSharpness, METH_VARARGS},
    {"hbr_set_edge_sharpness", HbrSetEdgeSharpness, METH_VARARGS},

    {NULL}
};

PyMODINIT_FUNC
initshim(void)
{
    PyObject* m = Py_InitModule3(
        "shim",
        osdFreeFunctions,
        "Python bindings for Pixar's OpenSubdiv library");

    OpaqueHbrMesh_Type.ob_type = &PyType_Type;
    OpaqueHbrMesh_Type.tp_new = OpaqueHbrMesh_new;
    OpaqueHbrMesh_Type.tp_methods = &OpaqueHbrMesh_methods[0];
    if (PyType_Ready(&OpaqueHbrMesh_Type) < 0) {
        printf("Can't register OpaqueHbrMesh type");
        return;
    }
    Py_INCREF(&OpaqueHbrMesh_Type);
    PyModule_AddObject(m, "OpaqueHbrMesh", (PyObject *)&OpaqueHbrMesh_Type);

    CSubd_Type.ob_type = &PyType_Type;
    CSubd_Type.tp_new = CSubd_new;
    CSubd_Type.tp_methods = &CSubd_methods[0];
    if (PyType_Ready(&CSubd_Type) < 0) {
        printf("Can't register CSubd type");
        return;
    }
    Py_INCREF(&CSubd_Type);
    PyModule_AddObject(m, "CSubd", (PyObject *)&CSubd_Type);

    // Numpy one-time initializations:
    import_array();

    // OSD one-time initializations:
    g_osdComputeController = new OpenSubdiv::OsdCpuComputeController();
}
