//
//     Copyright (C) Pixar. All rights reserved.
//
//     This license governs use of the accompanying software. If you
//     use the software, you accept this license. If you do not accept
//     the license, do not use the software.
//
//     1. Definitions
//     The terms "reproduce," "reproduction," "derivative works," and
//     "distribution" have the same meaning here as under U.S.
//     copyright law.  A "contribution" is the original software, or
//     any additions or changes to the software.
//     A "contributor" is any person or entity that distributes its
//     contribution under this license.
//     "Licensed patents" are a contributor's patent claims that read
//     directly on its contribution.
//
//     2. Grant of Rights
//     (A) Copyright Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free copyright license to reproduce its contribution,
//     prepare derivative works of its contribution, and distribute
//     its contribution or any derivative works that you create.
//     (B) Patent Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free license under its licensed patents to make, have
//     made, use, sell, offer for sale, import, and/or otherwise
//     dispose of its contribution in the software or derivative works
//     of the contribution in the software.
//
//     3. Conditions and Limitations
//     (A) No Trademark License- This license does not grant you
//     rights to use any contributor's name, logo, or trademarks.
//     (B) If you bring a patent claim against any contributor over
//     patents that you claim are infringed by the software, your
//     patent license from such contributor to the software ends
//     automatically.
//     (C) If you distribute any portion of the software, you must
//     retain all copyright, patent, trademark, and attribution
//     notices that are present in the software.
//     (D) If you distribute any portion of the software in source
//     code form, you may do so only under this license by including a
//     complete copy of this license with your distribution. If you
//     distribute any portion of the software in compiled or object
//     code form, you may only do so under a license that complies
//     with this license.
//     (E) The software is licensed "as-is." You bear the risk of
//     using it. The contributors give no express warranties,
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
//

%module shim

%{
#include "subdivider.h"
#include "topology.h"
%}

%header %{
#include <numpy/arrayobject.h>
#include <iostream>

static shim::DataType
_ShimTypeFromNumpyType(int numpyType)
{
    using namespace std;
    switch (numpyType) {
    case PyArray_FLOAT32: return shim::float32;
    case PyArray_UINT32:  return shim::uint32;
    case PyArray_INT32:   return shim::int32;
    case PyArray_INT8:    return shim::int8;
    case PyArray_UBYTE:   return shim::uint8;
    case PyArray_VOID: 
        cerr << "Expected a type but got VOID." << endl;
        abort();
        return shim::invalid;
    case PyArray_STRING: 
    case PyArray_UNICODE:
        cerr << "Strings are not expected here." << endl;
        return shim::invalid;
    case PyArray_OBJECT:
        cerr << "Complex type not allowed here." << endl;
        return shim::invalid;
    }
    cerr << "Unsupported numpy type " << numpyType << endl;
    abort();
    return shim::invalid;
}

static shim::DataType
_ShimTypeFromNumpyType(PyTypeObject* item)
{
    std::string numpyType = ((PyTypeObject*) item)->tp_name;
    if (numpyType == "numpy.float32") return shim::float32;
    if (numpyType == "numpy.uint32") return shim::uint32;
    if (numpyType == "numpy.int32") return shim::int32;
    if (numpyType == "numpy.uint8") return shim::uint8;
    std::cerr << "Unsupported numpy type " << numpyType << std::endl;
    return shim::invalid;
}

static shim::Layout
_ShimLayoutFromObject(PyObject* obj)
{
    using namespace std;
    shim::Layout layout;
    if (!obj) {
        cerr << "Missing type.\n";
    } else if (PyArray_DescrCheck(obj)) {
        PyArray_Descr* descr = (PyArray_Descr*) obj;
        int numpyType = descr->type_num;
        if (numpyType != PyArray_VOID) {
            shim::DataType dtype = _ShimTypeFromNumpyType(numpyType);
            layout.push_back(dtype);
        } else if (PyDataType_HASFIELDS(descr)) {
            PyObject* rawTypes = PyDict_Values(descr->fields);
            Py_ssize_t count = PySequence_Size(rawTypes);

            // Each field value is a type-offset pair.  We don't honor
            // the offsets, so just extract the type and move on.
            for (Py_ssize_t i = 0; i < count; ++i) {
                PyObject* pair = PySequence_ITEM(rawTypes, i);
                PyObject* item = PySequence_ITEM(pair, 0);
                shim::Layout newTypes = _ShimLayoutFromObject(item);
                layout.insert(layout.end(), newTypes.begin(), newTypes.end());
                Py_DECREF(item);
                Py_DECREF(pair);
            }

            Py_DECREF(rawTypes);
        } else {
            cerr << "Please provide a record-style numpy type with fields." << endl;
        }
    } else if (PyType_Check(obj)) { 
        layout.push_back( _ShimTypeFromNumpyType((PyTypeObject*) obj) );
    } else if (PySequence_Check(obj)) {
        Py_ssize_t count = PySequence_Size(obj);
        for (Py_ssize_t i = 0; i < count; ++i) {
            PyObject* item = PySequence_ITEM(obj, i);
            shim::Layout newTypes = _ShimLayoutFromObject(item);
            layout.insert(layout.end(), newTypes.begin(), newTypes.end());
            Py_DECREF(item);
        }
    } else {
        int dtype = (int) PyInt_AsLong(obj);
        if (dtype == -1) {
            cerr << "Got a " << obj->ob_type->tp_name << " but wanted a data type." << endl;
        } else if (dtype != shim::invalid) {
            layout.push_back(_ShimTypeFromNumpyType(dtype));
        }
    }
    return layout;
}

%}

%init %{
    import_array();
%}

%typemap(in) const shim::HomogeneousBuffer& {
    if (!PyArray_CheckExact($input)) {
        std::cerr << "This requires a numpy array." << std::endl;
    }
    $1 = new shim::HomogeneousBuffer();
    int byteCount = PyArray_NBYTES($input);
    unsigned char* begin = (unsigned char*) PyArray_BYTES($input);
    unsigned char* end = begin + byteCount;
    $1->Buffer.assign(begin, end);
    int dtype = PyArray_TYPE($input);
    $1->Type = _ShimTypeFromNumpyType(dtype);
}

%typemap(in) shim::Layout {
    $1 = _ShimLayoutFromObject($input);
}

%typemap(in) shim::DataType {
    using namespace std;
    shim::Layout layout = _ShimLayoutFromObject($input);
    if (layout.size() < 1) {
        cerr << "Can't convert PyObject " << $input << " to a shim data type." << endl;
        $1 = shim::invalid;
    } else if (layout.size() > 1) {
        cerr << "Complex data type not allowed here." << endl;
        $1 = shim::invalid;
    } else {
        $1 = layout[0];
    }
}

%typemap(in) const shim::HeterogeneousBuffer& {
    using namespace std;
    if (!PyArray_CheckExact($input)) {
        cerr << "This requires a numpy array." << endl;
    }
    $1 = new shim::HeterogeneousBuffer();
    int byteCount = PyArray_NBYTES($input);
    unsigned char* begin = (unsigned char*) PyArray_BYTES($input);
    unsigned char* end = begin + byteCount;
 
    $1->Buffer.assign(begin, end);
    $1->Layout = _ShimLayoutFromObject((PyObject*) PyArray_DESCR($input));
}

%typemap(in) shim::Buffer* INOUT {
    $1 = new shim::Buffer();
}

%typemap(argout) shim::Buffer* INOUT {
    using namespace std;
    PyObject* obj = (PyObject*) PyArray_DESCR($input);
    shim::Layout layout = _ShimLayoutFromObject(obj);
    if (layout.size() == 0) {
        cerr << "Unknown type for Buffer processing." << endl;
    } else if (layout[0] == shim::float32) {
        npy_intp numFloats = (npy_intp) ($1->size() / sizeof(float));
        float* ptrFloats = (float*) &(*$1)[0];

        $result = PyArray_SimpleNew(1, &numFloats, PyArray_FLOAT32);
        memcpy(PyArray_BYTES($result), ptrFloats, $1->size());

        // SimpleNewFromData would avoid a copy but we'd need to keep
        // the buffer around in our shim class.  It's not
        // impossible...
        //$result = PyArray_SimpleNewFromData(
        //    1, &numFloats, PyArray_FLOAT, (void*) ptrFloats);

    } else if (layout[0] == shim::uint32) {
        npy_intp numInts = (npy_intp) ($1->size() / sizeof(unsigned int));
        unsigned int* ptrInts = (unsigned int*) &(*$1)[0];
        $result = PyArray_SimpleNew(1, &numInts, PyArray_UINT32);
        memcpy(PyArray_BYTES($result), ptrInts, $1->size());
    } else {
        cerr << layout[0] << " is not an understood shim type."
             << endl;
    }
}

%typemap(freearg) shim::Buffer* INOUT {
    delete $1;
}

%include "subdivider.h"
%include "topology.h"
