//
//   Copyright 2013 Pixar
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

#ifndef OSD_DRAW_CONTEXT_H
#define OSD_DRAW_CONTEXT_H

#include "../version.h"

#include "../far/patchTables.h"

#include <utility>
#include <string>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

/// \brief Base DrawContext class
///
/// DrawContext derives several sub-classes with API specific functionality
/// (GL, D3D11, ...).
///
/// Current specificiation GPU hardware tessellation limitations require transition
/// patches to be split-up into several triangular bi-cubic sub-patches.
/// DrawContext processes FarPatchArrays from Far::PatchTables and generates the
/// additional sets of sub-patches.
///
/// Contexts interface the serialized topological data pertaining to the
/// geometric primitives with the capabilities of the selected discrete
/// compute device.
///
class DrawContext {

public:

    class PatchDescriptor {
    public:
        /// Constructor
        ///
        /// @param farDesc      Patch type descriptor
        ///
        /// @param maxValence   Highest vertex valence in the primitive
        ///
        /// @param subPatch     Index of the triangulated sub-patch for the given
        ///                     transition pattern. Transition patches need to be
        ///                     split into multiple sub-patches in order to be
        ///                     rendered with hardware tessellation.
        ///
        /// @param numElements  The size of the vertex and varying data per-vertex
        ///                     (in floats)
        ///
        PatchDescriptor(Far::PatchTables::Descriptor farDesc, unsigned char maxValence,
                    unsigned char subPatch, unsigned char numElements) :
            _farDesc(farDesc), _maxValence(maxValence), _subPatch(subPatch), _numElements(numElements) { }


        /// Returns the type of the patch
        Far::PatchTables::Type GetType() const {
            return _farDesc.GetType();
        }

        /// Returns the transition pattern of the patch if any (5 types)
        Far::PatchTables::TransitionPattern GetPattern() const {
            return _farDesc.GetPattern();
        }

        /// Returns the rotation of the patch (4 rotations)
        unsigned char GetRotation() const {
            return _farDesc.GetRotation();
        }

        /// Returns the number of control vertices expected for a patch of the
        /// type described
        int GetNumControlVertices() const {
            return _farDesc.GetNumControlVertices();
        }

        /// Returns the max valence
        int GetMaxValence() const {
            return _maxValence;
        }

        /// Returns the subpatch id
        int GetSubPatch() const {
            return _subPatch;
        }

        /// Returns the number of vertex elements
        int GetNumElements() const {
            return _numElements;
        }

        /// Set the number of vertex elements
        void SetNumElements(int numElements) {
            _numElements = (unsigned char)numElements;
        }

        /// Allows ordering of patches by type
        bool operator < ( PatchDescriptor const other ) const;

        /// True if the descriptors are identical
        bool operator == ( PatchDescriptor const other ) const;

    private:
        Far::PatchTables::Descriptor _farDesc;
        unsigned char _maxValence;
        unsigned char _subPatch;
        unsigned char _numElements;
    };

    class PatchArray {
    public:
        /// Constructor
        ///
        /// @param desc   Patch descriptor defines the type, pattern, rotation of
        ///               the patches in the array
        ///
        /// @param range  The range of vertex indices
        ///
        PatchArray(PatchDescriptor desc, Far::PatchTables::PatchArray::ArrayRange const & range) :
            _desc(desc), _range(range) { }

        /// Returns a patch descriptor defining the type of patches in the array
        PatchDescriptor GetDescriptor() const {
            return _desc;
        }

        /// Update a patch descriptor
        void SetDescriptor(PatchDescriptor desc) {
            _desc = desc;
        }

        /// Returns a array range struct
        Far::PatchTables::PatchArray::ArrayRange const & GetArrayRange() const {
            return _range;
        }

        /// Returns the index of the first control vertex of the first patch
        /// of this array in the global PTable
        unsigned int GetVertIndex() const {
            return _range.vertIndex;
        }

        /// Returns the global index of the first patch in this array (Used to
        /// access ptex / fvar table data)
        unsigned int GetPatchIndex() const {
            return _range.patchIndex;
        }

        /// Returns the number of patches in the array
        unsigned int GetNumPatches() const {
            return _range.npatches;
        }

        /// Returns the number of patch indices in the array
        unsigned int GetNumIndices() const {
            return _range.npatches * _desc.GetNumControlVertices();
        }

        /// Returns the offset of quad offset table
        unsigned int GetQuadOffsetIndex() const {
            return _range.quadOffsetIndex;
        }

        /// Set num patches (used at batch glomming)
        void SetNumPatches(int npatches) {
            _range.npatches = npatches;
        }

    private:
        PatchDescriptor _desc;
        Far::PatchTables::PatchArray::ArrayRange _range;
    };

    /// Constructor
    DrawContext() : _isAdaptive(false) {}

    /// Descrtuctor
    virtual ~DrawContext();

    /// Returns true if the primitive attached to the context uses feature adaptive
    /// subdivision
    bool IsAdaptive() const {
        return _isAdaptive;
    }

    typedef std::vector<PatchArray> PatchArrayVector;

    PatchArrayVector const & GetPatchArrays() const {
        return _patchArrays;
    }

    // processes FarPatchArrays and inserts requisite sub-patches for the arrays
    // containing transition patches
    static void ConvertPatchArrays(Far::PatchTables::PatchArrayVector const &farPatchArrays,
                                   DrawContext::PatchArrayVector &osdPatchArrays,
                                   int maxValence, int numElements);


    typedef std::vector<float> FVarData;

protected:

     static void packFVarData(Far::PatchTables const & patchTables,
                              int fvarWidth, FVarData const & src, FVarData & dst);

    // XXXX: move to private member
    PatchArrayVector _patchArrays;

    bool _isAdaptive;
};

// Allows ordering of patches by type
inline bool
DrawContext::PatchDescriptor::operator < ( PatchDescriptor const other ) const
{
    return _farDesc < other._farDesc or (_farDesc == other._farDesc and
          (_subPatch < other._subPatch or ((_subPatch == other._subPatch) and
          (_maxValence < other._maxValence or ((_maxValence == other._maxValence) and
          (_numElements < other._numElements))))));
}

// True if the descriptors are identical
inline bool
DrawContext::PatchDescriptor::operator == ( PatchDescriptor const other ) const
{
    return _farDesc == other._farDesc and
           _subPatch == other._subPatch and
           _maxValence == other._maxValence and
           _numElements == other._numElements;
}



}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_DRAW_CONTEXT_H */
