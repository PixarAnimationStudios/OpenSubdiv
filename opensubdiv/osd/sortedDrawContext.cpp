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

#include "../osd/sortedDrawContext.h"

#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdSortedDrawContext::OsdSortedDrawContext(FarPatchCountVector const &patchCounts,
                                           OsdPatchArrayVector const &patchArrays) {
    _patchCounts = patchCounts;
    _patchArrays = patchArrays;
    _patchDrawRangesDirty = true;
    _primFidelity.clear();
    _primFidelity.resize(patchCounts.size());
}

void
OsdSortedDrawContext::SetPrimFidelity(int primIndex, Fidelity f) {

    assert(primIndex >= 0 && primIndex <= (int)_primFidelity.size());
    _primFidelity[primIndex] = f;
    _patchDrawRangesDirty = true;
}

OsdPatchDrawRangeVector const &
OsdSortedDrawContext::GetPatchDrawRanges(OsdPatchDescriptor desc) {
    if (_patchDrawRangesDirty) {
        _ComputePatchDrawRanges();
        _patchDrawRangesDirty = false;
    }
    return _patchDrawRanges[desc];
}

void
OsdSortedDrawContext::_ComputePatchDrawRanges() {

    _patchDrawRanges.clear();

    for (size_t i = 0; i < _patchArrays.size(); ++i) {
        OsdPatchArray const &patch = _patchArrays[i];
        OsdPatchDescriptor desc = patch.desc;

        int offset = patch.firstIndex;
        for (size_t j = 0; j < _patchCounts.size(); ++j) {

            FarPatchCount const &counts = _patchCounts[j];
            int length = 0;
            switch (desc.type) {
            case kRegular:
                length = counts.regular*16;
                break;
            case kBoundary:
                length = counts.boundary*12;
                break;
            case kCorner:
                length = counts.corner*9;
                break;
            case kGregory:
                length = counts.gregory*4;
                break;
            case kBoundaryGregory:
                length = counts.boundaryGregory*4;
                break;
            case kTransitionRegular:
                length = counts.transitionRegular[desc.pattern]*16;
                break;
            case kTransitionBoundary:
                length = counts.transitionBoundary[desc.pattern][desc.rotation]*12;
                break;
            case kTransitionCorner:
                length = counts.transitionCorner[desc.pattern][desc.rotation]*9;
                break;
            }
            if (_primFidelity[j] != 0 and length > 0) {
                _patchDrawRanges[desc].push_back(OsdPatchDrawRange(offset, length));
            }
            offset += length;
        }
    }
}

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv


