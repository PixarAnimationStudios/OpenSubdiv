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
#ifndef VTR_SPARSE_SELECTOR_H
#define VTR_SPARSE_SELECTOR_H

#include "../version.h"

#include "../vtr/types.h"
#include "../vtr/refinement.h"

#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Vtr {

class Refinement;

//
//  SparseSelector:
//      This is experimental at present -- just keeping all of the functionality related to sparse
//  refinment out of Refinement for now until it matures.
//
//  Expected usage is as follows:
//
//          SparseSelector selector(refinement);
//
//          selector.selectFace(i);
//          selector.selectFace(j);
//          ...
//
//          //  To be later followed by:
//          refinement.refine(usingSparseSelectionOptions);
//
//  Since it is expected this will be protected or integrated elsewhere into another Vtr class --
//  which will be similarly protected -- all methods intentionally begin with lower case.
//
class SparseSelector {

public:
    SparseSelector(Refinement& refine) : _refine(&refine), _prevRefine(0), _selected(false) { }
    ~SparseSelector() { }

    //
    //  A previous refinement may be used to indicate whether components are fully defined or
    //  not -- note since optional it is specified/returned by pointer rather than reference
    //  (could make these both ptr for consistency...).
    //
    //  It is (increasingly) possible that this property ends up in the tags for the parent level,
    //  in which case this refinement that generated the parent will not be necessary
    //
    void           setRefinement(Refinement& refine) { _refine = &refine; }
    Refinement& getRefinement() const                { return *_refine; }

    void                 setPreviousRefinement(Refinement const* refine) { _prevRefine = refine; }
    Refinement const* getPreviousRefinement() const                      { return _prevRefine; }

    //
    //  Methods for selecting (and marking) components for refinement.  All component indices
    //  refer to components in the parent:
    //
    void selectVertex(Index pVertex);
    void selectEdge(  Index pEdge);
    void selectFace(  Index pFace);

    //  Mark all incident faces of a vertex -- common in the original feature-adaptive scheme
    //  to warrant inclusion, but may not be necessary if it is switch to being face-driven
    void selectVertexFaces(Index pVertex);

    //
    //  Useful queries during or after selection:
    //
    bool isSelectionEmpty() const { return !_selected; }

    bool isVertexIncomplete(Index pVertex) const {
        //  A parent of this refinement was child of the previous refinement:
        return _prevRefine && _prevRefine->_childVertexTag[pVertex]._incomplete;
    }

private:
    SparseSelector() { }

    bool wasVertexSelected(Index pVertex) const { return _refine->_parentVertexTag[pVertex]._selected; }
    bool wasEdgeSelected(  Index pEdge) const   { return _refine->_parentEdgeTag[pEdge]._selected; }
    bool wasFaceSelected(  Index pFace) const   { return _refine->_parentFaceTag[pFace]._selected; }

    void markVertexSelected(Index pVertex) const { _refine->_parentVertexTag[pVertex]._selected = true; }
    void markEdgeSelected(  Index pEdge) const   { _refine->_parentEdgeTag[pEdge]._selected = true; }
    void markFaceSelected(  Index pFace) const   { _refine->_parentFaceTag[pFace]._selected = true; }

    void markSelection();

private:
    Refinement*       _refine;
    Refinement const* _prevRefine;
    bool _selected;
};

} // end namespace Vtr

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;
} // end namespace OpenSubdiv

#endif /* VTR_SPARSE_SELECTOR_H */
