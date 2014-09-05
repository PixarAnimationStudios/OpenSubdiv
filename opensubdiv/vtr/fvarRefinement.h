//
//   Copyright 2014 DreamWorks Animation LLC.
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
#ifndef VTR_FVAR_REFINEMENT_H
#define VTR_FVAR_REFINEMENT_H

#include "../version.h"

#include "../sdc/type.h"
#include "../sdc/crease.h"
#include "../vtr/types.h"
#include "../vtr/refinement.h"
#include "../vtr/fvarLevel.h"

#include <vector>
#include <cassert>
#include <cstring>


namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Vtr {

//
//  FVarRefinement:
//      A face-varying refinement is the subset of face-varying data required to
//  support a "channel" of face-varying data.  Just as Refinement represents a
//  mapping between a parent and child Level, the face-varying analog represents
//  a mapping between a parent and child FVarLevel.
//
//  Its looking like this class may not be necessary...  Enough information exits
//  between the pair of FVarLevels and their Levels to do most of what we
//  want without needing to retain state information within the FVarRefinement as
//  is essential in Refinement (i.e. the parent-to-child mapping and vice versa).
//

class FVarRefinement {

public:
    FVarRefinement(Refinement const& refinement, FVarLevel& parent, FVarLevel& child);
    ~FVarRefinement();

    //  Const methods:
    Refinement const& getRefinement() const { return _refinement; }

    int getChildValueParentSource(Index vIndex, int sibling) const {
        return _childValueParentSource[_child->getVertexValueIndex(vIndex, (LocalIndex)sibling)];
    }

    //  Modifiers supporting application of the refinement:
    void applyRefinement();

    void estimateAndAllocateChildValues();
    void populateChildValues();
    int  populateChildValuesForEdgeVertex(Index cVert, Index pEdge, int offset);
    int  populateChildValuesForVertexVertex(Index cVert, Index pVert, int offset);
    void trimAndFinalizeChildValues();

    void propagateEdgeTags();
    void propagateValueTags();


public:
    //  The Refinement and familiar parent/child levels:
    Refinement const & _refinement;

    FVarLevel * _parent;
    FVarLevel * _child;

    //  When refinement is sparse, we need a mapping between siblings of a vertex
    //  value in the parent and child -- and for some child values, there will not
    //  be a parent value, in which case the source of the parent component will
    //  be stored.  So we refer to the parent "source" rather than "sibling":
    //
    std::vector<LocalIndex> _childValueParentSource;

    //
    //  These members are needed during refine() but currently serve no purpose
    //  after -- so we may not need this class persistent in the refine tables.
    //
    int _childSiblingFromEdgeCount;
    int _childSiblingFromVertCount;
};

} // end namespace Vtr

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;
} // end namespace OpenSubdiv

#endif /* VTR_FVAR_REFINEMENT_H */
