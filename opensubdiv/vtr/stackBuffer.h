//
//   Copyright 2015 DreamWorks Animation LLC.
//
//   Licensed under the Apache License, Version 2.0 (the "Apache License")
//   with the following modification; you may not use this file except in
//   compliance with the Apache License and the following modification to it:
//   Section 6. Trademarks. is deleted and replaced with:
//
//   6. Trademarks. This License does not grant permission to use the trade
//      names, trademarks, service marks, or product names of the Licensor
//      and its affiliates, except as required to comply with Section 4(c) of
//      the License and to reproduce the content of the NOTICE file.
//
//   You may obtain a copy of the Apache License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the Apache License with the above modification is
//   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//   KIND, either express or implied. See the Apache License for the specific
//   language governing permissions and limitations under the Apache License.
//
#ifndef OPENSUBDIV3_VTR_STACK_BUFFER_H
#define OPENSUBDIV3_VTR_STACK_BUFFER_H

#include "../version.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Vtr {
namespace internal {

//
//  The StackBuffer class is intented solely to take the place of VLA's (Variable
//  Length Arrays) which most compilers support, but are not strictly standard C++.
//  Portability concerns forces us to make use of either alloca() or some other
//  mechanism to create small arrays on the stack that are typically based on the
//  valence of a vertex -- small in general, but occassionally large.
//
//  Note also that since the intent of this is to replace VLA's -- not general
//  std::vectors -- support for std::vector functionality is intentionally limited
//  and STL-like naming is avoided.  Like a VLA there is no incremental growth.
//  Support for resizing is available to reuse an instance at the beginning of a
//  loop with a new size, but resizing in this case reinitializes all elements.
//

template <typename TYPE, unsigned int SIZE>
class StackBuffer
{
public:
    typedef unsigned int size_type;

public:
    //  Constructors and destructor -- declared inline below:
    StackBuffer();
    StackBuffer(size_type size);
    ~StackBuffer();

public:
    //  Note the reliance on implicit casting so that it can be used similar to
    //  a VLA.  This removes the need for operator[] as the resulting TYPE* will
    //  natively support [].  (The presence of both TYPE* and operator[] also
    //  causes an ambiguous overloading error with 32-bit MSVC builds.)

    operator TYPE const * () const { return _data; }
    operator TYPE *       ()       { return _data; }

    size_type GetSize() const { return _size; }

    void SetSize(size_type size);
    void Reserve(size_type capacity);

private:
    //  Non-copyable:
    StackBuffer(const StackBuffer<TYPE,SIZE> &) { }
    StackBuffer& operator=(const StackBuffer<TYPE,SIZE> &) { return *this; }

    void allocate(size_type capacity);
    void deallocate();
    void construct();
    void destruct();

private:
    TYPE *     _data;
    size_type  _size;
    size_type  _capacity;

    //  Is alignment an issue here?  The staticData arena will at least be double-word
    //  aligned within this struct, which meets current and most anticipated needs.
    char   _staticData[SIZE * sizeof(TYPE)];
    char * _dynamicData;
};


//
//  Core allocation/deallocation methods:
//
template <typename TYPE, unsigned int SIZE>
inline void
StackBuffer<TYPE,SIZE>::allocate(size_type capacity) {

    //  Again, is alignment an issue here?  C++ spec says new will return pointer
    //  "suitably aligned" for conversion to pointers of other types, which implies
    //  at least an alignment of 16.
    _dynamicData = static_cast<char*>(::operator new(capacity * sizeof(TYPE)));

    _data = reinterpret_cast<TYPE*>(_dynamicData);
    _capacity = capacity;
}

template <typename TYPE, unsigned int SIZE>
inline void
StackBuffer<TYPE,SIZE>::deallocate() {

    ::operator delete(_dynamicData);

    _data = reinterpret_cast<TYPE*>(_staticData);
    _capacity = SIZE;
}

//
//  Explicit element-wise construction and destruction within allocated memory (we
//  rely on the compiler to remove this code for types with empty constructors):
//
template <typename TYPE, unsigned int SIZE>
inline void
StackBuffer<TYPE,SIZE>::construct() {

    for (size_type i = 0; i < _size; ++i) {
        (void) new (&_data[i]) TYPE;
    }
}
template <typename TYPE, unsigned int SIZE>
inline void
StackBuffer<TYPE,SIZE>::destruct() {

    for (size_type i = 0; i < _size; ++i) {
        _data[i].~TYPE();
    }
}

//
//  Inline constructors and destructor:
//
template <typename TYPE, unsigned int SIZE>
inline
StackBuffer<TYPE,SIZE>::StackBuffer() :
    _data(reinterpret_cast<TYPE*>(_staticData)),
    _size(0),
    _capacity(SIZE),
    _dynamicData(0) {

}

template <typename TYPE, unsigned int SIZE>
inline
StackBuffer<TYPE,SIZE>::StackBuffer(size_type size) :
    _data(reinterpret_cast<TYPE*>(_staticData)),
    _size(size),
    _capacity(SIZE),
    _dynamicData(0) {

    if (size > SIZE) {
        allocate(size);
    }
    construct();
}

template <typename TYPE, unsigned int SIZE>
inline
StackBuffer<TYPE,SIZE>::~StackBuffer() {

    destruct();
    deallocate();
}

//
//  Inline sizing methods:
//
template <typename TYPE, unsigned int SIZE>
inline void
StackBuffer<TYPE,SIZE>::Reserve(size_type capacity) {

    if (capacity > _capacity) {
        destruct();
        deallocate();
        allocate(capacity);
    }
}

template <typename TYPE, unsigned int SIZE>
inline void
StackBuffer<TYPE,SIZE>::SetSize(size_type size)
{
    destruct();
    if (size == 0) {
        deallocate();
    } else if (size > _capacity) {
        deallocate();
        allocate(size);
    }
    _size = size;
    construct();
}

} // end namespace internal
} // end namespace Vtr

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;
} // end namespace OpenSubdiv

#endif /* OPENSUBDIV3_VTR_STACK_BUFFER_H */
