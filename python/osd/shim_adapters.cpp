static PyObject *
HbrUpdateFaces(PyObject *self, PyObject *args)
{
    OpaqueHbrMesh* hmesh = (OpaqueHbrMesh*) PyTuple_GetItem(args, 0);
    hmesh->faces->clear();
    hmesh->hmesh->GetFaces(std::back_inserter(*(hmesh->faces)));
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *
HbrGetVertexSharpness(PyObject *self, PyObject *args)
{
    OpaqueHbrMesh* hmesh = (OpaqueHbrMesh*) PyTuple_GetItem(args, 0);
    int index = (int) PyInt_AsLong(PyTuple_GetItem(args, 1));
    float sharpness = hmesh->hmesh->GetVertex(index)->GetSharpness();
    PyObject* retval = PyFloat_FromDouble(sharpness);
    Py_INCREF(retval);
    return retval;
}

static PyObject *
HbrSetVertexSharpness(PyObject *self, PyObject *args)
{
    OpaqueHbrMesh* hmesh = (OpaqueHbrMesh*) PyTuple_GetItem(args, 0);
    int index = (int) PyInt_AsLong(PyTuple_GetItem(args, 1));
    double value = (double) PyFloat_AsDouble(PyTuple_GetItem(args, 2));
    hmesh->hmesh->GetVertex(index)->SetSharpness(value);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *
HbrGetNumFaces(PyObject *self, PyObject *args)
{
    OpaqueHbrMesh* hmesh = (OpaqueHbrMesh*) PyTuple_GetItem(args, 0);
    PyObject* retval = PyInt_FromLong(hmesh->hmesh->GetNumFaces());
    Py_INCREF(retval);
    return retval;
}

static PyObject *
HbrGetFaceHole(PyObject *self, PyObject *args)
{
    OpaqueHbrMesh* hmesh = (OpaqueHbrMesh*) PyTuple_GetItem(args, 0);
    int faceIndex = (int) PyInt_AsLong(PyTuple_GetItem(args, 1));
    bool isHole = hmesh->faces->at(faceIndex)->IsHole();
    PyObject* retval = isHole ? Py_True : Py_False;
    Py_INCREF(retval);
    return retval;
}

static PyObject *
HbrSetFaceHole(PyObject *self, PyObject *args)
{
    OpaqueHbrMesh* hmesh = (OpaqueHbrMesh*) PyTuple_GetItem(args, 0);
    int faceIndex = (int) PyInt_AsLong(PyTuple_GetItem(args, 1));
    bool value = PyTuple_GetItem(args, 2) == Py_True;
    if (!value) {
        printf("Um, there's no API to unset a hole.\n");
    }
    
    hmesh->faces->at(faceIndex)->SetHole();
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *
HbrGetNumEdges(PyObject *self, PyObject *args)
{
    OpaqueHbrMesh* hmesh = (OpaqueHbrMesh*) PyTuple_GetItem(args, 0);
    int faceIndex = (int) PyInt_AsLong(PyTuple_GetItem(args, 1));
    int edgeCount = hmesh->faces->at(faceIndex)->GetNumVertices();
    PyObject* retval = PyInt_FromLong(edgeCount);
    Py_INCREF(retval);
    return retval;
}

static PyObject *
HbrGetEdgeSharpness(PyObject *self, PyObject *args)
{
    OpaqueHbrMesh* hmesh = (OpaqueHbrMesh*) PyTuple_GetItem(args, 0);
    int faceIndex = (int) PyInt_AsLong(PyTuple_GetItem(args, 1));
    int edgeIndex = (int) PyInt_AsLong(PyTuple_GetItem(args, 2));
    float sharpness = hmesh->faces->at(faceIndex)->
        GetEdge(edgeIndex)->GetSharpness();
    PyObject* retval = PyFloat_FromDouble(sharpness);
    Py_INCREF(retval);
    return retval;
}

static PyObject *
HbrSetEdgeSharpness(PyObject *self, PyObject *args)
{
    OpaqueHbrMesh* hmesh = (OpaqueHbrMesh*) PyTuple_GetItem(args, 0);
    int faceIndex = (int) PyInt_AsLong(PyTuple_GetItem(args, 1));
    int edgeIndex = (int) PyInt_AsLong(PyTuple_GetItem(args, 2));
    double value = (double) PyFloat_AsDouble(PyTuple_GetItem(args, 3));
    hmesh->faces->at(faceIndex)->GetEdge(edgeIndex)->SetSharpness(value);
    Py_INCREF(Py_None);
    return Py_None;
}
