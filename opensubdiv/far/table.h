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
#ifndef FAR_TABLE_H
#define FAR_TABLE_H

#include "../version.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

/// \brief A generic "table" with markers.
///
/// Generic multi-level indexing table : the indices across all the subdivision
/// levels are stored in a flat std::vector. 
///
/// The table class also holds a sequence of markers pointing to the first index
/// at the beginning of the sequence describing a given level.
/// (note that "level 1" vertices are obtained by using the indices starting at
/// "level 0" of the tables)
///
template <typename Type> class FarTable {
    std::vector<Type>   _data;     // table data
    std::vector<Type *> _markers;  // pointers to the first datum at each level
public:

    FarTable() { }

    FarTable(int maxlevel) : _markers(maxlevel) { }

    /// Reset max level and clear data
    void SetMaxLevel(int maxlevel) {
        _data.clear();
        _markers.resize(maxlevel);
    }

    /// Returns the memory required to store the data in this table.
    int GetMemoryUsed() const {
        return (int)_data.size() * sizeof(Type);
    }

    /// Returns the number of elements in level "level"
    int GetNumElements(int level) const {
        assert(level>=0 and level<((int)_markers.size()-1));
        return (int)(_markers[level+1] - _markers[level]);
    }

    /// Saves a pointer indicating the beginning of data pertaining to "level"
    /// of subdivision
    void SetMarker(int level, Type * marker) {
        _markers[level] = marker;
    }

    /// Resize the table to size (also resets markers)
    void Resize(int size) {
        _data.resize(size);
        _markers[0] = &_data[0];
    }

    /// Returns a pointer to the data at the beginning of level "level" of
    /// subdivision
    Type * operator[](int level) {
        assert(level>=0 and level<(int)_markers.size());
        return _markers[level];
    }

    /// Returns a const pointer to the data at the beginning of level "level"
    /// of subdivision
    const Type * operator[](int level) const {
        return const_cast<FarTable *>(this)->operator[](level);
    }
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_TABLE_H */
