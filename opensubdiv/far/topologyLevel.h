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
#ifndef OPENSUBDIV3_FAR_TOPOLOGY_LEVEL_H
#define OPENSUBDIV3_FAR_TOPOLOGY_LEVEL_H

#include "../version.h"

#include "../vtr/level.h"
#include "../vtr/refinement.h"
#include "../far/types.h"

#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

///
///  \brief TopologyLevel is an interface for accessing data in a specific level of a refined
///  topology hierarchy.  TopologyLevels are created and owned by a TopologyRefiner, which will
///  return const-references to them.  Such references are only valid during the lifetime of
///  TopologyRefiner that created and returned them, and only for a given refinement, i.e. if
///  the TopologyRefiner is re-refined, any references to TopoologyLevels are invalidated.
///
class TopologyLevel {

public:
    //  Inventory of components:
    int GetNumVertices() const     { return _level->getNumVertices(); }
    int GetNumEdges() const        { return _level->getNumEdges(); }
    int GetNumFaces() const        { return _level->getNumFaces(); }
    int GetNumFaceVertices() const { return _level->getNumFaceVerticesTotal(); }

    float GetEdgeSharpness(Index e) const   { return _level->getEdgeSharpness(e); }
    float GetVertexSharpness(Index v) const { return _level->getVertexSharpness(v); }
    bool  IsFaceHole(Index f) const         { return _level->isFaceHole(f); }

    Sdc::Crease::Rule GetVertexRule(Index v) const { return _level->getVertexRule(v); }

    int             GetNumFVarValues(          int channel = 0) const { return _level->getNumFVarValues(channel); }
    ConstIndexArray GetFVarFaceValues(Index f, int channel = 0) const { return _level->getFVarFaceValues(f, channel); }

    ConstIndexArray GetFaceVertices(Index f) const { return _level->getFaceVertices(f); }
    ConstIndexArray GetFaceEdges(Index f) const    { return _level->getFaceEdges(f); }
    ConstIndexArray GetEdgeVertices(Index e) const { return _level->getEdgeVertices(e); }
    ConstIndexArray GetEdgeFaces(Index e) const    { return _level->getEdgeFaces(e); }
    ConstIndexArray GetVertexFaces( Index v) const { return _level->getVertexFaces(v); }
    ConstIndexArray GetVertexEdges( Index v) const { return _level->getVertexEdges(v); }

    ConstLocalIndexArray GetEdgeFaceLocalIndices(Index e) const   { return _level->getEdgeFaceLocalIndices(e); }
    ConstLocalIndexArray GetVertexFaceLocalIndices(Index v) const { return _level->getVertexFaceLocalIndices(v); }
    ConstLocalIndexArray GetVertexEdgeLocalIndices(Index v) const { return _level->getVertexEdgeLocalIndices(v); }

    Index FindEdge(Index v0, Index v1) const { return _level->findEdge(v0, v1); }

    ConstIndexArray GetFaceChildFaces(Index f) const { return _refToChild->getFaceChildFaces(f); }
    ConstIndexArray GetFaceChildEdges(Index f) const { return _refToChild->getFaceChildEdges(f); }
    ConstIndexArray GetEdgeChildEdges(Index e) const { return _refToChild->getEdgeChildEdges(e); }

    Index GetFaceChildVertex(  Index f) const { return _refToChild->getFaceChildVertex(f); }
    Index GetEdgeChildVertex(  Index e) const { return _refToChild->getEdgeChildVertex(e); }
    Index GetVertexChildVertex(Index v) const { return _refToChild->getVertexChildVertex(v); }

    Index GetFaceParentFace(Index f) const { return _refToParent->getChildFaceParentFace(f); }

    bool ValidateTopology() const { return _level->validateTopology(); }
    void PrintTopology(bool children = true) const { _level->print((children && _refToChild) ? _refToChild : 0); }


private:
    friend class TopologyRefiner;

    Vtr::internal::Level const *      _level;
    Vtr::internal::Refinement const * _refToParent;
    Vtr::internal::Refinement const * _refToChild;

public:
    //  Not intended for public use, but required by std::vector, etc...
    TopologyLevel() { }
    ~TopologyLevel() { }
};

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;
} // end namespace OpenSubdiv

#endif /* OPENSUBDIV3_FAR_TOPOLOGY_LEVEL_H */
