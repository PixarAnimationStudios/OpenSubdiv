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

#ifndef OSD_DRAW_CONTEXT_H
#define OSD_DRAW_CONTEXT_H

#include "../version.h"
#include "../far/patchTables.h"

#include <utility>
#include <string>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdDrawContext {
public:
    class PatchDescriptor {
    public:
        PatchDescriptor(FarPatchTables::Descriptor farDesc, unsigned char maxValence,
                    unsigned char subPatch, unsigned char numElements) :
            _farDesc(farDesc), _maxValence(maxValence), _subPatch(subPatch), _numElements(numElements) { }


        /// Returns the type of the patch
        FarPatchTables::Type GetType() const {
            return _farDesc.GetType();
        }

        /// Returns the transition pattern of the patch if any (5 types)
        FarPatchTables::TransitionPattern GetPattern() const {
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
            _numElements = numElements;
        }

        /// Allows ordering of patches by type
        bool operator < ( PatchDescriptor const other ) const;

        /// True if the descriptors are identical
        bool operator == ( PatchDescriptor const other ) const;

    private:
        FarPatchTables::Descriptor _farDesc;
        unsigned char _maxValence;
        unsigned char _subPatch;
        unsigned char _numElements;
    };

    class PatchArray {
    public:
        PatchArray(PatchDescriptor desc, FarPatchTables::PatchArray::ArrayRange const & range) :
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
        FarPatchTables::PatchArray::ArrayRange const & GetArrayRange() const {
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
        FarPatchTables::PatchArray::ArrayRange _range;
    };

    typedef std::vector<PatchArray> PatchArrayVector;

    OsdDrawContext() : _isAdaptive(false) {}
    virtual ~OsdDrawContext();

    bool IsAdaptive() const { return _isAdaptive; }

    static void ConvertPatchArrays(FarPatchTables::PatchArrayVector const &farPatchArrays,
                                   OsdDrawContext::PatchArrayVector &osdPatchArrays,
                                   int maxValence, int numElements);

public:  // XXX: move to private member
    PatchArrayVector patchArrays;

protected:
    void createPatchArrays(FarPatchTables const * patchTables, int numElements);

    bool _isAdaptive;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_DRAW_CONTEXT_H */
