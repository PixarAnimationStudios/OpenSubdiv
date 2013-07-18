//
//     Copyright 2013 Pixar
//
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License
//     and the following modification to it: Section 6 Trademarks.
//     deleted and replaced with:
//
//     6. Trademarks. This License does not grant permission to use the
//     trade names, trademarks, service marks, or product names of the
//     Licensor and its affiliates, except as required for reproducing
//     the content of the NOTICE file.
//
//     You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//     Unless required by applicable law or agreed to in writing,
//     software distributed under the License is distributed on an
//     "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
//     either express or implied.  See the License for the specific
//     language governing permissions and limitations under the
//     License.
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
