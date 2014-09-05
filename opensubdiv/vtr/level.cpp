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
#include "../sdc/type.h"
#include "../sdc/crease.h"
#include "../vtr/array.h"
#include "../vtr/level.h"
#include "../vtr/refinement.h"
#include "../vtr/fvarLevel.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <map>


//
//  Level:
//      This is intended to be a fairly simple container of topology, sharpness and
//  other information that is useful to retain for subdivision.  It is intended to
//  be constructed by other friend classes, i.e. factories and class specialized to
//  contruct topology based on various splitting schemes.  So its interface consists
//  of simple methods for inspection, and low-level protected methods for populating
//  it rather than high-level modifiers.
//
namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Vtr {

//
//  Simple (for now) constructor and destructor:
//
Level::Level() :
    _faceCount(0),
    _edgeCount(0),
    _vertCount(0),
    _depth(0),
    _maxEdgeFaces(0),
    _maxValence(0) {
}

Level::~Level() {
    for (int i = 0; i < (int)_fvarChannels.size(); ++i) {
        delete _fvarChannels[i];
    }
}


//
//  Debugging method to validate topology, i.e. verify appropriate symmetry
//  between the relations, etc.
//
//  Additions that need to be made in the near term:
//      * verifying user-applied tags relating to topology:
//          - non-manifold in particular (ordering above can be part of this)
//          - face holes don't require anything
//      - verifying orientation of components, particularly vert-edges and faces:
//          - both need to be ordered correctly (when manifold)
//          - both need to be in sync for an interior vertex
//              ? is a rotation allowed for the interior case?
//              - I don't see why not...
//      ? verifying sharpness:
//          - values < Smooth or > Infinite
//          - sharpening of boundary edges (is this necessary, since we do it?)
//              - it does ensure our work was not corrupted by client assignments
//
//  Possibilities:
//      - single validate() method, which will call all of:
//          - validateTopology()
//          - validateSharpness()
//          - validateTagging()
//      - consider using a mask/struct to choose what to validate, i.e.:
//          - bool validate(ValidateOptions const& options) const;
//
bool
Level::validateTopology() const {

    //
    //  Verify internal topological consistency (eventually a Level method?):
    //      - each face-vert has corresponding vert-face (and child)
    //      - each face-edge has corresponding edge-face
    //      - each edge-vert has corresponding vert-edge (and child)
    //  The above three are enough for most cases, but it is still possible
    //  the latter relation in each above has no correspondent in the former,
    //  so apply the symmetric tests:
    //      - each edge-face has corresponding face-edge
    //      - each vert-face has corresponding face-vert
    //      - each vert-edge has corresponding edge-vert
    //  We are still left with the possibility of duplicate references in
    //  places we don't want them.  Currently a component can exist multiple
    //  times in a component of higher dimension.
    //      - each vert-face <face,child> pair is unique
    //      - each vert-edge <edge,child> pair is unique
    //
    bool returnOnFirstError = true;
    bool isValid = true;

    //  Verify each face-vert has corresponding vert-face and child:
    if ((getNumFaceVerticesTotal() == 0) || (getNumVertexFacesTotal() == 0)) {
        if (getNumFaceVerticesTotal() == 0) printf("Error:  missing face-verts\n");
        if (getNumVertexFacesTotal() == 0) printf("Error:  missing vert-faces\n");
        return false;
    }
    for (int fIndex = 0; fIndex < getNumFaces(); ++fIndex) {
        IndexArray const fVerts      = getFaceVertices(fIndex);
        int                 fVertCount  = fVerts.size();

        for (int i = 0; i < fVertCount; ++i) {
            Index vIndex = fVerts[i];

            IndexArray const      vFaces = getVertexFaces(vIndex);
            LocalIndexArray const vInFace = getVertexFaceLocalIndices(vIndex);

            bool vertFaceOfFaceExists = false;
            for (int j = 0; j < vFaces.size(); ++j) {
                if ((vFaces[j] == fIndex) && (vInFace[j] == i)) {
                    vertFaceOfFaceExists = true;
                    break;
                }
            }
            if (!vertFaceOfFaceExists) {
                isValid = false;
                if (returnOnFirstError) {
                    printf("Error in fIndex = %d:  correlation of vert %d failed\n", fIndex, i);
                    return isValid;
                }
            }
        }
    }

    //  Verify each face-edge has corresponding edge-face:
    if ((getNumEdgeFacesTotal() == 0) || (getNumFaceEdgesTotal() == 0)) {
        if (getNumEdgeFacesTotal() == 0) printf("Error:  missing edge-faces\n");
        if (getNumFaceEdgesTotal() == 0) printf("Error:  missing face-edges\n");
        return false;
    }
    for (int fIndex = 0; fIndex < getNumFaces(); ++fIndex) {
        IndexArray const fEdges      = getFaceEdges(fIndex);
        int                 fEdgeCount  = fEdges.size();

        for (int i = 0; i < fEdgeCount; ++i) {
            int eIndex = fEdges[i];

            IndexArray const eFaces      = getEdgeFaces(eIndex);
            int                 eFaceCount  = eFaces.size();

            bool edgeFaceOfFaceExists = false;
            for (int j = 0; j < eFaceCount; ++j) {
                if (eFaces[j] == fIndex) {
                    edgeFaceOfFaceExists = true;
                    break;
                }
            }
            if (!edgeFaceOfFaceExists) {
                isValid = false;
                if (returnOnFirstError) {
                    printf("Error in fIndex = %d:  correlation of edge %d failed\n", fIndex, i);
                    return isValid;
                }
            }
        }
    }

    //  Verify each edge-vert has corresponding vert-edge and child:
    if ((getNumEdgeVerticesTotal() == 0) || (getNumVertexEdgesTotal() == 0)) {
        if (getNumEdgeVerticesTotal() == 0) printf("Error:  missing edge-verts\n");
        if (getNumVertexEdgesTotal() == 0) printf("Error:  missing vert-edges\n");
        return false;
    }
    for (int eIndex = 0; eIndex < getNumEdges(); ++eIndex) {
        IndexArray const eVerts = getEdgeVertices(eIndex);

        for (int i = 0; i < 2; ++i) {
            Index vIndex = eVerts[i];

            IndexArray const      vEdges = getVertexEdges(vIndex);
            LocalIndexArray const vInEdge = getVertexEdgeLocalIndices(vIndex);

            bool vertEdgeOfEdgeExists = false;
            for (int j = 0; j < vEdges.size(); ++j) {
                if ((vEdges[j] == eIndex) && (vInEdge[j] == i)) {
                    vertEdgeOfEdgeExists = true;
                    break;
                }
            }
            if (!vertEdgeOfEdgeExists) {
                isValid = false;
                if (returnOnFirstError) {
                    printf("Error in eIndex = %d:  correlation of vert %d failed\n", eIndex, i);
                    return isValid;
                }
            }
        }
    }

    //  Verify non-manifold tags are appropriately assigned to edges and vertices:
    //      - note we have to validate orientation of vertex neighbors to do this rigorously
    for (int eIndex = 0; eIndex < getNumEdges(); ++eIndex) {
        Level::ETag const& eTag = _edgeTags[eIndex];
        if (eTag._nonManifold) continue;

        IndexArray const eVerts = getEdgeVertices(eIndex);
        if (eVerts[0] == eVerts[1]) {
            isValid = false;
            if (returnOnFirstError) {
                printf("Error in eIndex = %d:  degenerate edge not tagged marked non-manifold\n", eIndex);
                return isValid;
            }
        }

        IndexArray const eFaces = getEdgeFaces(eIndex);
        if ((eFaces.size() < 1) || (eFaces.size() > 2)) {
            isValid = false;
            if (returnOnFirstError) {
                printf("Error in eIndex = %d:  edge with %d faces not tagged non-manifold\n", eIndex, eFaces.size());
                return isValid;
            }
        }
    }
    return isValid;
}

//
//  Anonymous helper functions for debugging output -- yes, using printf(), this is not
//  intended to serve anyone other than myself for now and I favor its formatting control
//
namespace {
    template <typename INT_TYPE>
    void
    printIndexArray(Array<INT_TYPE> const& array) {

        printf("%d [%d", array.size(), array[0]);
        for (int i = 1; i < array.size(); ++i) {
            printf(" %d", array[i]);
        }
        printf("]\n");
    }

    const char*
    ruleString(Sdc::Crease::Rule rule) {
        switch (rule) {
            case Sdc::Crease::RULE_UNKNOWN: return "<uninitialized>";
            case Sdc::Crease::RULE_SMOOTH:  return "Smooth";
            case Sdc::Crease::RULE_DART:    return "Dart";
            case Sdc::Crease::RULE_CREASE:  return "Crease";
            case Sdc::Crease::RULE_CORNER:  return "Corner";
            default:
                assert(0);
        }
        return 0;
    }
}

void
Level::print(const Refinement* pRefinement) const {

    bool printFaceVerts      = true;
    bool printFaceEdges      = true;
    bool printFaceChildVerts = false;
    bool printFaceTags       = true;

    bool printEdgeVerts      = true;
    bool printEdgeFaces      = true;
    bool printEdgeChildVerts = true;
    bool printEdgeSharpness  = true;
    bool printEdgeTags       = true;

    bool printVertFaces      = true;
    bool printVertEdges      = true;
    bool printVertChildVerts = false;
    bool printVertSharpness  = true;
    bool printVertTags       = true;

    printf("Level (0x%p):\n", this);
    printf("  Depth = %d\n", _depth);

    printf("  Primary component counts:\n");
    printf("    faces = %d\n", _faceCount);
    printf("    edges = %d\n", _edgeCount);
    printf("    verts = %d\n", _vertCount);

    printf("  Topology relation sizes:\n");

    printf("    Face relations:\n");
    printf("      face-vert counts/offset = %lu\n", _faceVertCountsAndOffsets.size());
    printf("      face-vert indices = %lu\n", _faceVertIndices.size());
    for (int i = 0; printFaceVerts && i < getNumFaces(); ++i) {
        printf("        face %4d verts:  ", i);
        printIndexArray(getFaceVertices(i));
    }
    printf("      face-edge indices = %lu\n", _faceEdgeIndices.size());
    for (int i = 0; printFaceEdges && i < getNumFaces(); ++i) {
        printf("        face %4d edges:  ", i);
        printIndexArray(getFaceEdges(i));
    }
    printf("      face tags = %lu\n", _faceTags.size());
    for (int i = 0; printFaceTags && i < (int)_faceTags.size(); ++i) {
        FTag const& fTag = _faceTags[i];
        printf("        face %4d:", i);
        printf("  hole = %d",  (int)fTag._hole);
        printf("\n");
    }
    if (pRefinement) {
        printf("      face child-verts = %lu\n", pRefinement->_faceChildVertIndex.size());
        for (int i = 0; printFaceChildVerts && i < (int)pRefinement->_faceChildVertIndex.size(); ++i) {
            printf("        face %4d child vert:  %d\n", i, pRefinement->_faceChildVertIndex[i]);
        }
    }

    printf("    Edge relations:\n");
    printf("      edge-vert indices = %lu\n", _edgeVertIndices.size());
    for (int i = 0; printEdgeVerts && i < getNumEdges(); ++i) {
        printf("        edge %4d verts:  ", i);
        printIndexArray(getEdgeVertices(i));
    }
    printf("      edge-face counts/offset = %lu\n", _edgeFaceCountsAndOffsets.size());
    printf("      edge-face indices = %lu\n", _edgeFaceIndices.size());
    for (int i = 0; printEdgeFaces && i < getNumEdges(); ++i) {
        printf("        edge %4d faces:  ", i);
        printIndexArray(getEdgeFaces(i));
    }
    if (pRefinement) {
        printf("      edge child-verts = %lu\n", pRefinement->_edgeChildVertIndex.size());
        for (int i = 0; printEdgeChildVerts && i < (int)pRefinement->_edgeChildVertIndex.size(); ++i) {
            printf("        edge %4d child vert:  %d\n", i, pRefinement->_edgeChildVertIndex[i]);
        }
    }
    printf("      edge sharpness = %lu\n", _edgeSharpness.size());
    for (int i = 0; printEdgeSharpness && i < (int)_edgeSharpness.size(); ++i) {
        printf("        edge %4d sharpness:  %f\n", i, _edgeSharpness[i]);
    }
    printf("      edge tags = %lu\n", _edgeTags.size());
    for (int i = 0; printEdgeTags && i < (int)_edgeTags.size(); ++i) {
        ETag const& eTag = _edgeTags[i];
        printf("        edge %4d:", i);
        printf("  boundary = %d",  (int)eTag._boundary);
        printf(", semiSharp = %d", (int)eTag._semiSharp);
        printf(", infSharp = %d",  (int)eTag._infSharp);
        printf("\n");
    }

    printf("    Vert relations:\n");
    printf("      vert-face counts/offset = %lu\n", _vertFaceCountsAndOffsets.size());
    printf("      vert-face indices  = %lu\n", _vertFaceIndices.size());
    printf("      vert-face children = %lu\n", _vertFaceLocalIndices.size());
    for (int i = 0; printVertFaces && i < getNumVertices(); ++i) {
        printf("        vert %4d faces:  ", i);
        printIndexArray(getVertexFaces(i));

        printf("             face-verts:  ");
        printIndexArray(getVertexFaceLocalIndices(i));
    }
    printf("      vert-edge counts/offset = %lu\n", _vertEdgeCountsAndOffsets.size());
    printf("      vert-edge indices  = %lu\n", _vertEdgeIndices.size());
    printf("      vert-edge children = %lu\n", _vertEdgeLocalIndices.size());
    for (int i = 0; printVertEdges && i < getNumVertices(); ++i) {
        printf("        vert %4d edges:  ", i);
        printIndexArray(getVertexEdges(i));

        printf("             edge-verts:  ");
        printIndexArray(getVertexEdgeLocalIndices(i));
    }
    if (pRefinement) {
        printf("      vert child-verts = %lu\n", pRefinement->_vertChildVertIndex.size());
        for (int i = 0; printVertChildVerts && i < (int)pRefinement->_vertChildVertIndex.size(); ++i) {
            printf("        vert %4d child vert:  %d\n", i, pRefinement->_vertChildVertIndex[i]);
        }
    }
    printf("      vert sharpness = %lu\n", _vertSharpness.size());
    for (int i = 0; printVertSharpness && i < (int)_vertSharpness.size(); ++i) {
        printf("        vert %4d sharpness:  %f\n", i, _vertSharpness[i]);
    }
    printf("      vert tags = %lu\n", _vertTags.size());
    for (int i = 0; printVertTags && i < (int)_vertTags.size(); ++i) {
        VTag const& vTag = _vertTags[i];
        printf("        vert %4d:", i);
        printf("  rule = %s",      ruleString((Sdc::Crease::Rule)vTag._rule));
        printf(", boundary = %d",  (int)vTag._boundary);
        printf(", xordinary = %d", (int)vTag._xordinary);
        printf(", semiSharp = %d", (int)vTag._semiSharp);
        printf(", infSharp = %d",  (int)vTag._infSharp);
        printf("\n");
    }
    fflush(stdout);
}


namespace {
    template <typename TAG_TYPE, typename INT_TYPE>
    void
    combineTags(TAG_TYPE& dstTag, TAG_TYPE const& srcTag) {
        INT_TYPE const* srcInt = reinterpret_cast<INT_TYPE const*>(&srcTag);
        INT_TYPE *      dstInt = reinterpret_cast<INT_TYPE *>     (&dstTag);

        *dstInt |= *srcInt;
    }
}

Level::VTag
Level::getFaceCompositeVTag(IndexArray const& faceVerts) const {

    VTag compTag = _vertTags[faceVerts[0]];

    for (int i = 1; i < faceVerts.size(); ++i) {
        VTag const& vertTag = _vertTags[faceVerts[i]];

        if (sizeof(VTag) == sizeof(unsigned short)) {
            combineTags<VTag, unsigned short>(compTag, vertTag);
        } else {
            assert("VTag size is uint_32 -- need to adjust composite tag code..." == 0);
        }
    }
    return compTag;
}


//
//  High-level topology gathering functions -- used mainly in patch construction.  These
//  may eventually be moved elsewhere, possibly to classes specialized for quad- and tri-
//  patch identification and construction, but for now somewhere more accessible than the
//  patch tables factory is preferable.
//
//  Note a couple of nuisances...
//      - these are currently specialized methods for quad-meshes
//      - debatable whether we should include the four face-verts in the face functions
//          - we refer to the result as a "patch" when we do
//          - otherwise a "ring" of vertices is more appropriate
//      - some OSD containers for the results want unsigned int and others int
//
namespace {
    template <typename INT_TYPE>
    inline INT_TYPE fastMod4(INT_TYPE value) { return (value & 0x3); }

    inline int
    fastFindIn4(Index value, IndexArray const& array) {

        if (value == array[0]) return 0;
        if (value == array[1]) return 1;
        if (value == array[2]) return 2;
        if (value == array[3]) return 3;
        assert("fastFindIn4() did not find expected value!" == 0);
        return -1;
    }
}

//
//  Gathering the one-ring of vertices from quads surrounding a manifold vertex:
//      - the neighborhood of the vertex is assumed to be quad-regular
//
//  Ordering of resulting vertices:
//      The surrounding one-ring follows the ordering of the incident faces.  For each
//  incident quad, the two vertices in CCW order within that quad are added.  If the
//  vertex is on a boundary, a third vertex on the boundary edge will be contributed from
//  the last face.
//
int
Level::gatherManifoldVertexRingFromIncidentQuads(Index vIndex, int vOffset, int ringVerts[]) const {

    Level const& level = *this;

    IndexArray vEdges = level.getVertexEdges(vIndex);

    IndexArray      vFaces   = level.getVertexFaces(vIndex);
    LocalIndexArray vInFaces = level.getVertexFaceLocalIndices(vIndex);

    bool isBoundary = (vEdges.size() > vFaces.size());

    int ringIndex = 0;
    for (int i = 0; i < vFaces.size(); ++i) {
        //
        //  For every incident quad, we want the two vertices clockwise in each face, i.e.
        //  the vertex at the end of the leading edge and the vertex opposite this one:
        //
        IndexArray fVerts = level.getFaceVertices(vFaces[i]);

        int vInThisFace = vInFaces[i];

        ringVerts[ringIndex++] = vOffset + fVerts[fastMod4(vInThisFace + 1)];
        ringVerts[ringIndex++] = vOffset + fVerts[fastMod4(vInThisFace + 2)];

        if (isBoundary && (i == (vFaces.size() - 1))) {
            ringVerts[ringIndex++] = vOffset + fVerts[fastMod4(vInThisFace + 3)];
        }
    }
    return ringIndex;
}

//
//  Gathering the 16 vertices of a quad-regular boundary patch:
//      - the neighborhood of the face is assumed to be quad-regular
//
//  Ordering of resulting vertices:
//      It was debatable whether to include the vertices of the original face for a complete
//  "patch" or just the surrounding ring -- clearly we ended up with a function for the entire
//  patch, but that may change.
//      The latter ring of vertices around the face (potentially returned on its own) was
//  oriented with respect to the face.  The ring of 12 vertices is gathered as 4 groups of 3
//  vertices -- one for each corner vertex, and each group forming the quad opposite each
//  corner vertex when combined with that corner vertex.  The four vertices of the face begin
//  the patch.
//
//         |     |     |     |
//      ---5-----4-----15----14---
//         |     |     |     |
//         |     |     |     |
//      ---6-----0-----3-----13---
//         |     |x   x|     |
//         |     |x   x|     |
//      ---7-----1-----2-----12---
//         |     |     |     |
//         |     |     |     |
//      ---8-----9-----10----11---
//         |     |     |     |
//
int
Level::gatherQuadRegularInteriorPatchVertices(
    Index thisFace, unsigned int ringVerts[], int rotation) const {

    Level const& level = *this;

    //
    //  For each of the four corner vertices, there is a face diagonally opposite
    //  the given/central face, within which are three vertices of the ring:
    //
    IndexArray thisFaceVerts = level.getFaceVertices(thisFace);
    if (rotation) {
        ringVerts[0] = thisFaceVerts[fastMod4(rotation)];
        ringVerts[1] = thisFaceVerts[fastMod4(rotation + 1)];
        ringVerts[2] = thisFaceVerts[fastMod4(rotation + 2)];
        ringVerts[3] = thisFaceVerts[fastMod4(rotation + 3)];
    } else {
        ringVerts[0] = thisFaceVerts[0];
        ringVerts[1] = thisFaceVerts[1];
        ringVerts[2] = thisFaceVerts[2];
        ringVerts[3] = thisFaceVerts[3];
    }

    int ringIndex = 4;
    for (int i = 0; i < 4; ++i) {
        Index v = ringVerts[i];

        IndexArray      vFaces   = level.getVertexFaces(v);
        LocalIndexArray vInFaces = level.getVertexFaceLocalIndices(v);

        int thisFaceInVFaces = fastFindIn4(thisFace, vFaces);
        int intFaceInVFaces  = fastMod4(thisFaceInVFaces + 2);

        Index intFace    = vFaces[intFaceInVFaces];
        int      vInIntFace = vInFaces[intFaceInVFaces];

        IndexArray intFaceVerts = level.getFaceVertices(intFace);

        ringVerts[ringIndex++] = intFaceVerts[fastMod4(vInIntFace + 1)];
        ringVerts[ringIndex++] = intFaceVerts[fastMod4(vInIntFace + 2)];
        ringVerts[ringIndex++] = intFaceVerts[fastMod4(vInIntFace + 3)];
    }
    assert(ringIndex == 16);
    return 16;
}

//
//  Gathering the 12 vertices of a quad-regular boundary patch:
//      - the neighborhood of the face is assumed to be quad-regular
//      - the single edge of the face that lies on the boundary is specified
//      - only one edge of the face is a boundary edge
//
//  Ordering of resulting vertices:
//      It was debatable whether to include the vertices of the original face for a complete
//  "patch" or just the surrounding ring -- clearly we ended up with a function for the entire
//  patch, but that may change.
//      The latter ring of vertices around the face (potentially returned on its own) was
//  oriented beginning from the leading CCW boundary edge and ending at the trailing edge.
//  The four vertices of the face begin the patch and are oriented similarly to this outer
//  ring -- forming an inner ring that begins and ends in the same manner.
//
//      ---4-----0-----3-----11---
//         |     |x   x|     |
//         |     |x   x|     |
//      ---5-----1-----2-----10---
//         |     |     |     |
//         |     |     |     |
//      ---6-----7-----8-----9----
//         |     |     |     |
//
int
Level::gatherQuadRegularBoundaryPatchVertices(
    Index face, int unsigned ringVerts[], int boundaryEdgeInFace) const {

    Level const& level = *this;

    int interiorEdgeInFace = fastMod4(boundaryEdgeInFace + 2);

    //
    //  V0 and V1 are the two interior vertices (opposite the boundary edge) around
    //  which we will gather most of the ring:
    //
    int intV0InFace = interiorEdgeInFace;
    int intV1InFace = fastMod4(interiorEdgeInFace + 1);

    IndexArray faceVerts = level.getFaceVertices(face);

    Index v0 = faceVerts[intV0InFace];
    Index v1 = faceVerts[intV1InFace];

    IndexArray v0Faces = level.getVertexFaces(v0);
    IndexArray v1Faces = level.getVertexFaces(v1);

    LocalIndexArray v0InFaces = level.getVertexFaceLocalIndices(v0);
    LocalIndexArray v1InFaces = level.getVertexFaceLocalIndices(v1);

    int boundaryFaceInV0Faces = -1;
    int boundaryFaceInV1Faces = -1;
    for (int i = 0; i < 4; ++i) {
        if (face == v0Faces[i]) boundaryFaceInV0Faces = i;
        if (face == v1Faces[i]) boundaryFaceInV1Faces = i;
    }
    assert((boundaryFaceInV0Faces >= 0) && (boundaryFaceInV1Faces >= 0));

    //  Identify the four faces of interest -- previous to and opposite V0 and
    //  opposite and next from V1 -- relative to V0 and V1:
    int prevFaceInV0Faces = fastMod4(boundaryFaceInV0Faces + 1);
    int intFaceInV0Faces  = fastMod4(boundaryFaceInV0Faces + 2);
    int intFaceInV1Faces  = fastMod4(boundaryFaceInV1Faces + 2);
    int nextFaceInV1Faces = fastMod4(boundaryFaceInV1Faces + 3);

    //  Identify the indices of the four faces:
    Index prevFace  = v0Faces[prevFaceInV0Faces];
    Index intV0Face = v0Faces[intFaceInV0Faces];
    Index intV1Face = v1Faces[intFaceInV1Faces];
    Index nextFace  = v1Faces[nextFaceInV1Faces];

    //  Identify V0 and V1 relative to these four faces:
    LocalIndex v0InPrevFace = v0InFaces[prevFaceInV0Faces];
    LocalIndex v0InIntFace  = v0InFaces[intFaceInV0Faces];
    LocalIndex v1InIntFace  = v1InFaces[intFaceInV1Faces];
    LocalIndex v1InNextFace = v1InFaces[nextFaceInV1Faces];

    //  Access the vertices of these four faces and assign to the ring:
    IndexArray prevFaceVerts  = level.getFaceVertices(prevFace);
    IndexArray intV0FaceVerts = level.getFaceVertices(intV0Face);
    IndexArray intV1FaceVerts = level.getFaceVertices(intV1Face);
    IndexArray nextFaceVerts  = level.getFaceVertices(nextFace);

    ringVerts[0] = faceVerts[fastMod4(boundaryEdgeInFace + 1)];
    ringVerts[1] = faceVerts[fastMod4(boundaryEdgeInFace + 2)];
    ringVerts[2] = faceVerts[fastMod4(boundaryEdgeInFace + 3)];
    ringVerts[3] = faceVerts[         boundaryEdgeInFace];

    ringVerts[4] = prevFaceVerts[fastMod4(v0InPrevFace + 2)];

    ringVerts[5] = intV0FaceVerts[fastMod4(v0InIntFace + 1)];
    ringVerts[6] = intV0FaceVerts[fastMod4(v0InIntFace + 2)];
    ringVerts[7] = intV0FaceVerts[fastMod4(v0InIntFace + 3)];

    ringVerts[8]  = intV1FaceVerts[fastMod4(v1InIntFace + 1)];
    ringVerts[9]  = intV1FaceVerts[fastMod4(v1InIntFace + 2)];
    ringVerts[10] = intV1FaceVerts[fastMod4(v1InIntFace + 3)];

    ringVerts[11] = nextFaceVerts[fastMod4(v1InNextFace + 2)];

    return 12;
}

//
//  Gathering the 9 vertices of a quad-regular corner patch:
//      - the neighborhood of the face is assumed to be quad-regular
//      - the single corner vertex is specified
//      - only one vertex of the face is a corner
//
//  Ordering of resulting vertices:
//      It was debatable whether to include the vertices of the original face for a complete
//  "patch" or just the surrounding ring -- clearly we ended up with a function for the entire
//  patch, but that may change.
//      Like the boundary case, the latter ring of vertices around the face was oriented
//  beginning from the leading CCW boundary edge and ending at the trailing edge.  The four
//  face vertices begin the patch, and begin with the corner vertex.
//
//      0-----3-----8---
//      |x   x|     |
//      |x   x|     |
//      1-----2-----7---
//      |     |     |
//      |     |     |
//      4-----5-----6---
//      |     |     |
//
int
Level::gatherQuadRegularCornerPatchVertices(
    Index face, unsigned int ringVerts[], int cornerVertInFace) const {

    Level const& level = *this;

    int interiorFaceVert = fastMod4(cornerVertInFace + 2);

    IndexArray faceVerts = level.getFaceVertices(face);
    Index      intVert   = faceVerts[interiorFaceVert];

    IndexArray      intVertFaces   = level.getVertexFaces(intVert);
    LocalIndexArray intVertInFaces = level.getVertexFaceLocalIndices(intVert);

    int cornerFaceInIntVertFaces = -1;
    for (int i = 0; i < intVertFaces.size(); ++i) {
        if (face == intVertFaces[i]) {
            cornerFaceInIntVertFaces = i;
            break;
        }
    }
    assert(cornerFaceInIntVertFaces >= 0);

    //  Identify the three faces relative to the interior vertex:
    int prevFaceInIntVertFaces = fastMod4(cornerFaceInIntVertFaces + 1);
    int intFaceInIntVertFaces  = fastMod4(cornerFaceInIntVertFaces + 2);
    int nextFaceInIntVertFaces = fastMod4(cornerFaceInIntVertFaces + 3);

    //  Identify the indices of the three other faces:
    Index prevFace = intVertFaces[prevFaceInIntVertFaces];
    Index intFace  = intVertFaces[intFaceInIntVertFaces];
    Index nextFace = intVertFaces[nextFaceInIntVertFaces];

    //  Identify the interior vertex relative to these three faces:
    LocalIndex intVertInPrevFace = intVertInFaces[prevFaceInIntVertFaces];
    LocalIndex intVertInIntFace  = intVertInFaces[intFaceInIntVertFaces];
    LocalIndex intVertInNextFace = intVertInFaces[nextFaceInIntVertFaces];

    //  Access the vertices of these three faces and assign to the ring:
    IndexArray prevFaceVerts = level.getFaceVertices(prevFace);
    IndexArray intFaceVerts  = level.getFaceVertices(intFace);
    IndexArray nextFaceVerts = level.getFaceVertices(nextFace);

    ringVerts[0] = faceVerts[         cornerVertInFace];
    ringVerts[1] = faceVerts[fastMod4(cornerVertInFace + 1)];
    ringVerts[2] = faceVerts[fastMod4(cornerVertInFace + 2)];
    ringVerts[3] = faceVerts[fastMod4(cornerVertInFace + 3)];

    ringVerts[4] = prevFaceVerts[fastMod4(intVertInPrevFace + 2)];

    ringVerts[5] = intFaceVerts[fastMod4(intVertInIntFace + 1)];
    ringVerts[6] = intFaceVerts[fastMod4(intVertInIntFace + 2)];
    ringVerts[7] = intFaceVerts[fastMod4(intVertInIntFace + 3)];

    ringVerts[8] = nextFaceVerts[fastMod4(intVertInNextFace + 2)];

    return 9;
}



//
//  What follows is an internal/anonymous class and protected methods to complete all
//  topological relations when only the face-vertex relations is defined.
//
//  In keeping with the original idea that Level is just data and relies on other
//  classes to construct it, this functionality may be warranted elsewhere, but we are
//  collectively unclear as to where that should be at present.  In the meantime, the
//  implementation is provided here so that we can test and make use of it.
//
namespace {
    //
    //  This is an internal helper class to manage the assembly of the tological relations
    //  that do not have a predictable size, i.e. faces-per-edge, faces-per-vertex and
    //  edges-per-vertex.  Level manages these with two vectors:
    //
    //      - a vector of integer pairs for the "counts" and "offsets"
    //      - a vector of incident members accessed by the "offset" of each
    //
    //  The "dynamic relation" allocates the latter vector of members based on a typical
    //  number of members per component, e.g. we expect valence 4 vertices in a typical
    //  quad-mesh, and so an "expected" number might be 6 to accomodate a few x-ordinary
    //  vertices.  The member vector is allocated with this number per component and the
    //  counts and offsets initialized to refer to them -- but with the counts set to 0.
    //  The count will be incremented as members are identified and entered, and if any
    //  component "overflows" the expected number of members, the members are moved to a
    //  separate vector in an std::map for the component.
    //
    //  Once all incident members have been added, the main vector is compressed and may
    //  need to merge entries from the map in the process.
    //
    typedef std::map<Index, IndexVector> IrregIndexMap;

    class DynamicRelation {
    public:
        DynamicRelation(IndexVector& countAndOffsets, IndexVector& indices, int membersPerComp);
        ~DynamicRelation() { }

    public:
        //  Methods dealing with the members for each component:
        IndexArray getCompMembers(Index index);
        void          appendCompMember(Index index, Index member);

        //  Methods dealing with the components:
        void appendComponent();
        void compressMemberIndices();

    public:
        int _compCount;
        int _memberCountPerComp;

        IndexVector & _countsAndOffsets;
        IndexVector & _regIndices;

        IrregIndexMap _irregIndices;
    };

    inline
    DynamicRelation::DynamicRelation(IndexVector& countAndOffsets, IndexVector& indices, int membersPerComp) :
            _compCount(0),
            _memberCountPerComp(membersPerComp),
            _countsAndOffsets(countAndOffsets),
            _regIndices(indices) {

        _compCount = (int) _countsAndOffsets.size() / 2;

        for (int i = 0; i < _compCount; ++i) {
            _countsAndOffsets[2*i]   = 0;
            _countsAndOffsets[2*i+1] = i * _memberCountPerComp;
        }
        _regIndices.resize(_compCount * _memberCountPerComp);
    }

    inline IndexArray
    DynamicRelation::getCompMembers(Index compIndex) {

        int count = _countsAndOffsets[2*compIndex];
        if (count > _memberCountPerComp) {
            IndexVector & irregMembers = _irregIndices[compIndex];
            return IndexArray(&irregMembers[0], (int)irregMembers.size());
        } else {
            int offset = _countsAndOffsets[2*compIndex+1];
            return IndexArray(&_regIndices[offset], count);
        }
    }
    inline void
    DynamicRelation::appendCompMember(Index compIndex, Index memberValue) {

        int count  = _countsAndOffsets[2*compIndex];
        int offset = _countsAndOffsets[2*compIndex+1];

        if (count < _memberCountPerComp) {
            _regIndices[offset + count] = memberValue;
        } else {
            IndexVector& irregMembers = _irregIndices[compIndex];

            if (count > _memberCountPerComp) {
                irregMembers.push_back(memberValue);
            } else {
                irregMembers.resize(_memberCountPerComp + 1);
                std::memcpy(&irregMembers[0], &_regIndices[offset], sizeof(Index) * _memberCountPerComp);
                irregMembers[_memberCountPerComp] = memberValue;
            }
        }
        _countsAndOffsets[2*compIndex] ++;
    }
    inline void
    DynamicRelation::appendComponent() {

        _countsAndOffsets.push_back(0);
        _countsAndOffsets.push_back(_compCount * _memberCountPerComp);

        ++ _compCount;
        _regIndices.resize(_compCount * _memberCountPerComp);
    }
    void
    DynamicRelation::compressMemberIndices() {

        if (_irregIndices.size() == 0) {
            int memberCount = _countsAndOffsets[0];
            for (int i = 1; i < _compCount; ++i) {
                int count  = _countsAndOffsets[2*i];
                int offset = _countsAndOffsets[2*i + 1];

                memmove(&_regIndices[memberCount], &_regIndices[offset], count * sizeof(Index));

                _countsAndOffsets[2*i + 1] = memberCount;
                memberCount += count;
            }
            _regIndices.resize(memberCount);
        } else {
            //  Assign new offsets-per-component while determining if we can trivially compressed in place:
            bool cannotBeCompressedInPlace = false;

            int memberCount = _countsAndOffsets[0];
            for (int i = 1; i < _compCount; ++i) {
                _countsAndOffsets[2*i + 1] = memberCount;

                cannotBeCompressedInPlace |= (memberCount > (_memberCountPerComp * i));

                memberCount += _countsAndOffsets[2*i];
            }
            cannotBeCompressedInPlace |= (memberCount > (_memberCountPerComp * _compCount));

            //  Copy members into the original or temporary vector accordingly:
            IndexVector  tmpIndices;
            if (cannotBeCompressedInPlace) {
                tmpIndices.resize(memberCount);
            }
            IndexVector& dstIndices = cannotBeCompressedInPlace ? tmpIndices : _regIndices;
            for (int i = 0; i < _compCount; ++i) {
                int count = _countsAndOffsets[2*i];

                Index *dstMembers = &dstIndices[_countsAndOffsets[2*i + 1]];
                Index *srcMembers = (count <= _memberCountPerComp)
                                     ? &_regIndices[i * _memberCountPerComp]
                                     : &_irregIndices[i][0];
                memmove(dstMembers, srcMembers, count * sizeof(Index));
            }
            if (cannotBeCompressedInPlace) {
                _regIndices.swap(tmpIndices);
            } else {
                _regIndices.resize(memberCount);
            }
        }
    }
}


//
//  Methods to populate the missing topology relations of the Level:
//
inline Index
Level::findEdge(Index v0Index, Index v1Index, IndexArray const& v0Edges) const {

    if (v0Index != v1Index) {
        for (int j = 0; j < v0Edges.size(); ++j) {
            IndexArray eVerts = this->getEdgeVertices(v0Edges[j]);
            if ((eVerts[0] == v1Index) || (eVerts[1] == v1Index)) {
                return v0Edges[j];
            }
        }
    } else {
        for (int j = 0; j < v0Edges.size(); ++j) {
            IndexArray eVerts = this->getEdgeVertices(v0Edges[j]);
            if (eVerts[0] == eVerts[1]) {
                return v0Edges[j];
            }
        }
    }
    return INDEX_INVALID;
}

Index
Level::findEdge(Index v0Index, Index v1Index) const {
    return this->findEdge(v0Index, v1Index, this->getVertexEdges(v0Index));
}

void
Level::completeTopologyFromFaceVertices() {

    //
    //  Its assumed (a pre-condition) that face-vertices have been fully specified and that we
    //  are to construct the remaining relations:  including the edge list.  We may want to
    //  support the existence of the edge list too in future:
    //
    int vCount = this->getNumVertices();
    int fCount = this->getNumFaces();
    int eCount = this->getNumEdges();
    assert((vCount > 0) && (fCount > 0) && (eCount == 0));

    //  May be unnecessary depending on how the vertices and faces were defined, but worth a
    //  call to ensure all data related to verts and faces is available -- this will be a
    //  harmless call if all has been taken care of).
    //
    //  Remember to resize edges similarly after the edge list has been assembled...
    this->resizeVertices(vCount);
    this->resizeFaces(fCount);
    this->resizeEdges(0);

    //
    //  Resize face-edges to match face-verts and reserve for edges based on an estimate:
    //
    this->_faceEdgeIndices.resize(this->getNumFaceVerticesTotal());

    int eCountEstimate = (vCount << 1);

    this->_edgeVertIndices.reserve(eCountEstimate * 2);
    this->_edgeFaceIndices.reserve(eCountEstimate * 2);

    this->_edgeFaceCountsAndOffsets.reserve(eCountEstimate * 2);

    //
    //  Create the dynamic relations to be populated (edge-faces will remain empty as reserved
    //  above since there are currently no edges) and iterate through the faces to do so:
    //
    const int avgSize = 6;

    DynamicRelation dynEdgeFaces(this->_edgeFaceCountsAndOffsets, this->_edgeFaceIndices, 2);
    DynamicRelation dynVertFaces(this->_vertFaceCountsAndOffsets, this->_vertFaceIndices, avgSize);
    DynamicRelation dynVertEdges(this->_vertEdgeCountsAndOffsets, this->_vertEdgeIndices, avgSize);

    for (Index fIndex = 0; fIndex < fCount; ++fIndex) {
        IndexArray fVerts = this->getFaceVertices(fIndex);
        IndexArray fEdges = this->getFaceEdges(fIndex);

        for (int i = 0; i < fVerts.size(); ++i) {
            Index v0Index = fVerts[i];
            Index v1Index = fVerts[(i+1) % fVerts.size()];

            //  Look for the edge in v0's incident edge members:
            IndexArray v0Edges = dynVertEdges.getCompMembers(v0Index);

            Index eIndex = this->findEdge(v0Index, v1Index, v0Edges);

            //  If no edge found, create/append a new one:
            if (!IndexIsValid(eIndex)) {
                eIndex = (Index) this->_edgeCount;

                this->_edgeCount ++;
                this->_edgeVertIndices.push_back(v0Index);
                this->_edgeVertIndices.push_back(v1Index);

                dynEdgeFaces.appendComponent();

                dynVertEdges.appendCompMember(v0Index, eIndex);
                dynVertEdges.appendCompMember(v1Index, eIndex);
            }
            dynEdgeFaces.appendCompMember(eIndex,  fIndex);
            dynVertFaces.appendCompMember(v0Index, fIndex);

            fEdges[i] = eIndex;
        }
        _maxValence = std::max(_maxValence, fVerts.size());
    }

    dynEdgeFaces.compressMemberIndices();
    dynVertFaces.compressMemberIndices();
    dynVertEdges.compressMemberIndices();

    //
    //  At this point all incident members are associated with each component.  We now need
    //  to populate the "local indices" for each -- accounting for on-manifold potential --
    //  and orient each set.  There is little wortwhile advantage in having the local indices
    //  available for the orientation as the orienting code "walks" around the components
    //  independent of their given order.  And since determining the local indices is more
    //  involved for non-manifold vertices (needing to deal with repeated entries) we are
    //  better of orienting to determine manifold status and then computing local indices
    //  according to the manifold status.
    //
    //  Resize edges with the Level to ensure anything else related to edges is created:
    eCount = this->getNumEdges();
    this->resizeEdges(eCount);

    for (Index eIndex = 0; eIndex < eCount; ++eIndex) {
        Level::ETag& eTag = this->_edgeTags[eIndex];

        IndexArray eFaces = this->getEdgeFaces(eIndex);
        IndexArray eVerts = this->getEdgeVertices(eIndex);

        _maxEdgeFaces = std::max(_maxEdgeFaces, eFaces.size());

        if ((eFaces.size() < 1) || (eFaces.size() > 2)) {
            eTag._nonManifold = true;
        }
        if (eVerts[0] == eVerts[1]) {
            printf("ASSERTION - degenerate edges not yet supported!\n");
            assert(eVerts[0] != eVerts[1]);

            eTag._nonManifold = true;
        }

        //  Mark incident vertices non-manifold to avoid attempting to orient them:
        if (eTag._nonManifold) {
            this->_vertTags[eVerts[0]]._nonManifold = true;
            this->_vertTags[eVerts[1]]._nonManifold = true;
        }
    }
    orientIncidentComponents();

    populateLocalIndices();
}

void
Level::populateLocalIndices() {

    //
    //  We have two sets of local indices -- vert-faces and vert-edges:
    //
    int vCount = this->getNumVertices();

    this->_vertFaceLocalIndices.resize(this->_vertFaceIndices.size());
    this->_vertEdgeLocalIndices.resize(this->_vertEdgeIndices.size());

    for (Index vIndex = 0; vIndex < vCount; ++vIndex) {
        IndexArray      vFaces   = this->getVertexFaces(vIndex);
        LocalIndexArray vInFaces = this->getVertexFaceLocalIndices(vIndex);

        for (int i = 0; i < vFaces.size(); ++i) {
            IndexArray fVerts = this->getFaceVertices(vFaces[i]);

            int vInFaceIndex = (int)(std::find(fVerts.begin(), fVerts.end(), vIndex) - fVerts.begin());
            vInFaces[i] = (LocalIndex) vInFaceIndex;
        }
    }

    for (Index vIndex = 0; vIndex < vCount; ++vIndex) {
        IndexArray      vEdges   = this->getVertexEdges(vIndex);
        LocalIndexArray vInEdges = this->getVertexEdgeLocalIndices(vIndex);

        for (int i = 0; i < vEdges.size(); ++i) {
            IndexArray eVerts = this->getEdgeVertices(vEdges[i]);

            vInEdges[i] = (vIndex == eVerts[1]);
        }
        _maxValence = std::max(_maxValence, vEdges.size());
    }
}

void
Level::orientIncidentComponents() {

    int vCount = this->getNumVertices();

    for (Index vIndex = 0; vIndex < vCount; ++vIndex) {
        Level::VTag vTag = this->_vertTags[vIndex];

        if (!vTag._nonManifold) {
            if (!orderVertexFacesAndEdges(vIndex)) {
                vTag._nonManifold = true;
            }
        }
    }
}

namespace {
    inline int
    findInArray(IndexArray const& array, Index value) {
        return (int)(std::find(array.begin(), array.end(), value) - array.begin());
    }
}

bool
Level::orderVertexFacesAndEdges(Index vIndex) {

    IndexArray vEdges = this->getVertexEdges(vIndex);
    IndexArray vFaces = this->getVertexFaces(vIndex);

    int fCount = vFaces.size();
    int eCount = vEdges.size();

    if ((fCount == 0) || (eCount < 2) || ((eCount - fCount) > 1)) return false;

    //
    //  Note we have already eliminated the possibility of incident degenerate edges
    //  and other bad edges earlier -- marking its vertices non-manifold as a result
    //  and explicitly avoiding this method:
    //
    Index fStart  = INDEX_INVALID;
    Index eStart  = INDEX_INVALID;
    int      fvStart = 0;

    if (eCount == fCount) {
        //  Interior case -- start with the first face

        fStart  = vFaces[0];
        fvStart = findInArray(this->getFaceVertices(fStart), vIndex);
        eStart  = this->getFaceEdges(fStart)[fvStart];
    } else {
        //  Boundary case -- start with (identify) the leading of two boundary edges:

        for (int i = 0; i < eCount; ++i) {
            IndexArray const eFaces = this->getEdgeFaces(vEdges[i]);
            if (eFaces.size() == 1) {
                eStart = vEdges[i];
                fStart = eFaces[0];
                fvStart = findInArray(this->getFaceVertices(fStart), vIndex);

                //  Singular edge -- look for forward edge to this vertex:
                if (eStart == (this->getFaceEdges(fStart)[fvStart])) {
                    break;
                }
            }
        }
    }

    //
    //  We have identified a starting face, face-vert and leading edge from
    //  which to walk counter clockwise to identify manifold neighbors.  If
    //  this vertex is really locally manifold, we will end up back at the
    //  starting edge or at the other singular edge of a boundary:
    //
    Index * vFacesOrdered = (Index *)alloca(fCount*sizeof(Index)),
             * vEdgesOrdered = (Index *)alloca(eCount*sizeof(Index));

    int orderedEdgesCount = 1;
    int orderedFacesCount = 1;

    vFacesOrdered[0] = fStart;
    vEdgesOrdered[0] = eStart;

    Index eFirst = eStart;

    while (orderedEdgesCount < eCount) {
        //
        //  Find the next edge, i.e. the one counter-clockwise to the last:
        //
        IndexArray const fVerts = this->getFaceVertices(fStart);
        IndexArray const fEdges = this->getFaceEdges(fStart);

        int      feStart = fvStart;
        int      feNext  = feStart ? (feStart - 1) : (fVerts.size() - 1);
        Index eNext   = fEdges[feNext];

        //  Two non-manifold situations detected:
        //      - two subsequent edges the same, i.e. a "repeated edge" in a face
        //      - back at the start before all edges processed
        if ((eNext == eStart) || (eNext == eFirst)) return false;

        //
        //  Add the next edge and if more faces to visit (not at the end of
        //  a boundary) look to its opposite face:
        //
        vEdgesOrdered[orderedEdgesCount++] = eNext;

        if (orderedFacesCount < fCount) {
            IndexArray const eFaces = this->getEdgeFaces(eNext);

            if (eFaces.size() == 0) return false;
            if ((eFaces.size() == 1) && (eFaces[0] == fStart)) return false;

            fStart  = eFaces[eFaces[0] == fStart];
            fvStart = findInArray(this->getFaceEdges(fStart), eNext);

            vFacesOrdered[orderedFacesCount++] = fStart;
        }
        eStart = eNext;
    }
    assert(orderedEdgesCount == eCount);
    assert(orderedFacesCount == fCount);

    //  All faces and edges have been ordered -- apply and return:
    std::memcpy(&vFaces[0], vFacesOrdered, fCount * sizeof(Index));
    std::memcpy(&vEdges[0], vEdgesOrdered, eCount * sizeof(Index));

    return true;
}

//
//  In development -- methods for accessing face-varying data channels...
//
int
Level::createFVarChannel(int fvarValueCount, Sdc::Options const& fvarOptions) {

    FVarLevel* fvarLevel = new FVarLevel(*this);

    fvarLevel->setOptions(fvarOptions);
    fvarLevel->resizeValues(fvarValueCount);
    fvarLevel->resizeComponents();

    _fvarChannels.push_back(fvarLevel);
    return (int)_fvarChannels.size() - 1;
}

void
Level::destroyFVarChannel(int channel) {

    delete _fvarChannels[channel];
    _fvarChannels.erase(_fvarChannels.begin() + channel);
}

int
Level::getNumFVarValues(int channel) const {
    return _fvarChannels[channel]->getNumValues();
}

IndexArray const
Level::getFVarFaceValues(Index faceIndex, int channel) const {
    return _fvarChannels[channel]->getFaceValues(faceIndex);
}

IndexArray
Level::getFVarFaceValues(Index faceIndex, int channel) {
    return _fvarChannels[channel]->getFaceValues(faceIndex);
}

void
Level::completeFVarChannelTopology(int channel) {
    return _fvarChannels[channel]->completeTopologyFromFaceValues();
}

} // end namespace Vtr

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
