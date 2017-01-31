//
//   Copyright 2015 Pixar
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

#include "../far/gregoryBasis.h"
#include "../far/endCapBSplineBasisPatchFactory.h"
#include "../far/error.h"
#include "../far/stencilTableFactory.h"
#include "../far/topologyRefiner.h"

#include <cassert>
#include <cmath>
#include <cstring>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

namespace {
#ifdef __INTEL_COMPILER
#pragma warning (push)
#pragma warning disable 1572
#endif
    inline bool isWeightNonZero(float w) { return (w != 0.0f); }
#ifdef __INTEL_COMPILER
#pragma warning (pop)
#endif
}

EndCapBSplineBasisPatchFactory::EndCapBSplineBasisPatchFactory(
    TopologyRefiner const & refiner,
    StencilTable * vertexStencils,
    StencilTable * varyingStencils) :
    _vertexStencils(vertexStencils), _varyingStencils(varyingStencils),
    _refiner(&refiner), _numVertices(0), _numPatches(0) {

    // Sanity check: the mesh must be adaptively refined
    assert(! refiner.IsUniform());

    // Reserve the patch point stencils. Ideally topology refiner
    // would have an API to return how many endcap patches will be required.
    // Instead we conservatively estimate by the number of patches at the
    // finest level.
    int numMaxLevelFaces = refiner.GetLevel(refiner.GetMaxLevel()).GetNumFaces();

    // we typically use 7 patch points for each bspline endcap.
    int numPatchPointsExpected = numMaxLevelFaces * 7;
    // limits to 100M (=800M bytes) entries for the reserved size.
    int numStencilsExpected = std::min(numPatchPointsExpected * 16,
                                       100*1024*1024);
    _vertexStencils->reserve(numPatchPointsExpected, numStencilsExpected);
    if (_varyingStencils) {
        // varying stencils use only 1 index with weight=1.0
        _varyingStencils->reserve(numPatchPointsExpected, numPatchPointsExpected);
    }
}

ConstIndexArray
EndCapBSplineBasisPatchFactory::GetPatchPoints(
    Vtr::internal::Level const * level, Index thisFace,
    Vtr::internal::Level::VSpan const cornerSpans[],
    int levelVertOffset, int fvarChannel) {

    //
    //  We can only use a faster method directly with B-Splines when we have a
    //  single interior irregular corner.  We defer to an intermediate Gregory
    //  patch in all other cases, i.e. the presence of any boundary, more than
    //  one irregular vertex or use of the partial neighborhood at any corner
    //  (not a true boundary wrt the corner vertex, but imposed by some other
    //  feature -- inf-sharp crease, face-varying discontinuity, etc).
    //
    //  Assume we don't need to use a Gregory patch until we identify a feature
    //  at any corner that indicates we do.
    //
    Vtr::ConstIndexArray facePoints = level->getFaceVertices(thisFace);

    int irregCornerIndex = -1;
    bool useGregoryPatch = (fvarChannel >= 0);

    for (int corner = 0; (corner < 4) && !useGregoryPatch; ++corner) {
        Vtr::internal::Level::VTag vtag = level->getVertexTag(facePoints[corner]);

        if ((vtag._rule != Sdc::Crease::RULE_SMOOTH) || cornerSpans[corner].isAssigned()) {
            useGregoryPatch = true;
        }
        if (vtag._xordinary) {
            if (irregCornerIndex < 0) {
                irregCornerIndex = corner;
            } else {
                useGregoryPatch = true;
            }
        }
    }

    if (useGregoryPatch) {
        return getPatchPointsFromGregoryBasis(
            level, thisFace, cornerSpans, facePoints,
            levelVertOffset, fvarChannel);
    } else {
        return getPatchPoints(
            level, thisFace, irregCornerIndex, facePoints,
            levelVertOffset, fvarChannel);
    }
}

ConstIndexArray
EndCapBSplineBasisPatchFactory::getPatchPointsFromGregoryBasis(
    Vtr::internal::Level const * level, Index thisFace,
    Vtr::internal::Level::VSpan const cornerSpans[],
    ConstIndexArray facePoints, int levelVertOffset, int fvarChannel) {

    // XXX: For now, always create new 16 indices for each patch.
    // we'll optimize later to share all regular control points with
    // other patches as well as try to make extra-ordinary verts watertight.

    int offset = (fvarChannel < 0)
               ? _refiner->GetNumVerticesTotal()
               : _refiner->GetNumFVarValuesTotal(fvarChannel);
    for (int i = 0; i < 16; ++i) {
        _patchPoints.push_back(_numVertices + offset);
        ++_numVertices;
    }
    GregoryBasis::ProtoBasis basis(*level, thisFace, cornerSpans, levelVertOffset, fvarChannel);
    // XXX: temporary hack. we should traverse topology and find existing
    //      vertices if available
    //
    // Reorder gregory basis stencils into regular bezier
    GregoryBasis::Point const *bezierCP[16];

    bezierCP[0] = &basis.P[0];
    bezierCP[1] = &basis.Ep[0];
    bezierCP[2] = &basis.Em[1];
    bezierCP[3] = &basis.P[1];

    bezierCP[4] = &basis.Em[0];
    bezierCP[5] = &basis.Fp[0]; // arbitrary
    bezierCP[6] = &basis.Fp[1]; // arbitrary
    bezierCP[7] = &basis.Ep[1];

    bezierCP[8]  = &basis.Ep[3];
    bezierCP[9]  = &basis.Fp[3]; // arbitrary
    bezierCP[10] = &basis.Fp[2]; // arbitrary
    bezierCP[11] = &basis.Em[2];

    bezierCP[12] = &basis.P[3];
    bezierCP[13] = &basis.Em[3];
    bezierCP[14] = &basis.Ep[2];
    bezierCP[15] = &basis.P[2];

    // all stencils should have the same capacity.
    int stencilCapacity = basis.P[0].GetCapacity();

    // Apply basis conversion from bezier to b-spline
    float Q[4][4] = {{ 6, -7,  2, 0},
                     { 0,  2, -1, 0},
                     { 0, -1,  2, 0},
                     { 0,  2, -7, 6} };
    Vtr::internal::StackBuffer<GregoryBasis::Point, 16> H(16);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            H[i*4+j].Clear(stencilCapacity);
            for (int k = 0; k < 4; ++k) {
                if (isWeightNonZero(Q[i][k])) {
                    H[i*4+j].AddWithWeight(*bezierCP[j+k*4], Q[i][k]);
                }
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            GregoryBasis::Point p(stencilCapacity);
            for (int k = 0; k < 4; ++k) {
                if (isWeightNonZero(Q[j][k])) {
                    p.AddWithWeight(H[i*4+k], Q[j][k]);
                }
            }
            GregoryBasis::AppendToStencilTable(p, _vertexStencils);
        }
    }
    if (_varyingStencils) {
        int varyingIndices[] = { 0, 0, 1, 1,
                                 0, 0, 1, 1,
                                 3, 3, 2, 2,
                                 3, 3, 2, 2,};
        for (int i = 0; i < 16; ++i) {
            int varyingIndex = facePoints[varyingIndices[i]] + levelVertOffset;
            _varyingStencils->_sizes.push_back(1);
            _varyingStencils->_indices.push_back(varyingIndex);
            _varyingStencils->_weights.push_back(1.0f);
        }
    }

    ++_numPatches;
    return ConstIndexArray(&_patchPoints[(_numPatches-1)*16], 16);
}

void
EndCapBSplineBasisPatchFactory::computeLimitStencils(
    Vtr::internal::Level const *level,
    ConstIndexArray facePoints, int vid, int fvarChannel,
    GregoryBasis::Point *P, GregoryBasis::Point *Ep, GregoryBasis::Point *Em)
{
    int maxvalence = level->getMaxValence();

    Vtr::internal::StackBuffer<Index, 40> manifoldRing;
    manifoldRing.SetSize(maxvalence*2);

    int ringSize =
        level->gatherQuadRegularRingAroundVertex(
            facePoints[vid], manifoldRing, fvarChannel);

    // note: this function has not yet supported boundary.
    assert((ringSize & 1) == 0);
    int valence = ringSize/2;
    int stencilCapacity = ringSize + 1;

    Index start = -1, prev = -1;
    {
        int ip = (vid+1)%4, im = (vid+3)%4;
        for (int i = 0; i < valence; ++i) {
            if (manifoldRing[i*2] == facePoints[ip])
                start = i;
            if (manifoldRing[i*2] == facePoints[im])
                prev = i;
        }
    }
    assert(start > -1 && prev > -1);

    GregoryBasis::Point e0, e1;
    e0.Clear(stencilCapacity);
    e1.Clear(stencilCapacity);

    float t = 2.0f * float(M_PI) / float(valence);
    float ef = 1.0f / (valence * (cosf(t) + 5.0f +
                                  sqrtf((cosf(t) + 9) * (cosf(t) + 1)))/16.0f);

    for (int i = 0; i < valence; ++i) {
        Index ip = (i+1)%valence;
        Index idx_neighbor   = (manifoldRing[2*i  + 0]),
              idx_diagonal   = (manifoldRing[2*i  + 1]),
              idx_neighbor_p = (manifoldRing[2*ip + 0]);

        float d = float(valence)+5.0f;

        GregoryBasis::Point f(4);
        f.AddWithWeight(facePoints[vid], float(valence)/d);
        f.AddWithWeight(idx_neighbor_p,  2.0f/d);
        f.AddWithWeight(idx_neighbor,    2.0f/d);
        f.AddWithWeight(idx_diagonal,    1.0f/d);

        P->AddWithWeight(f, 1.0f/float(valence));

        float c0 = 0.5f*cosf((float(2*M_PI) * float(i)/float(valence)))
                 + 0.5f*cosf((float(2*M_PI) * float(ip)/float(valence)));
        float c1 = 0.5f*sinf((float(2*M_PI) * float(i)/float(valence)))
                 + 0.5f*sinf((float(2*M_PI) * float(ip)/float(valence)));
        e0.AddWithWeight(f, c0*ef);
        e1.AddWithWeight(f, c1*ef);
    }

    *Ep = *P;
    Ep->AddWithWeight(e0, cosf((float(2*M_PI) * float(start)/float(valence))));
    Ep->AddWithWeight(e1, sinf((float(2*M_PI) * float(start)/float(valence))));

    *Em = *P;
    Em->AddWithWeight(e0, cosf((float(2*M_PI) * float(prev)/float(valence))));
    Em->AddWithWeight(e1, sinf((float(2*M_PI) * float(prev)/float(valence))));
}

ConstIndexArray
EndCapBSplineBasisPatchFactory::getPatchPoints(
    Vtr::internal::Level const *level, Index thisFace,
    Index extraOrdinaryIndex, ConstIndexArray facePoints,
    int levelVertOffset, int fvarChannel) {

    //  Fast B-spline endcap construction.
    //
    //  This function assumes the patch is not on boundary
    //  and it contains only 1 extraordinary vertex.
    //  The location of the extraordinary vertex can be one of
    //  0-ring quad corner.
    //
    //  B-Spline control point gathering indices
    //
    //     [5]   (4)---(15)--(14)    0 : extraordinary vertex
    //            |     |     |
    //            |     |     |      1,2,3,9,10,11,12,13 :
    //     (6)----0-----3-----13       B-Spline control points, gathered by
    //      |     |     |     |         traversing topology
    //      |     |     |     |
    //     (7)----1-----2-----12     (5) :
    //      |     |     |     |        Fitted patch point (from limit position)
    //      |     |     |     |
    //     (8)----9-----10----11     (4),(6),(7),(8),(14),(15) :
    //                                 Fitted patch points
    //                                   (from limit tangents and bezier CP)
    //
    static int const rotation[4][16] = {
        /*= 0 ring =*/ /* ================ 1 ring ================== */
        { 0, 1, 2, 3,    4,  5,  6,  7,  8,  9, 10, 11, 12, 13 ,14, 15},
        { 1, 2, 3, 0,    7,  8,  9, 10, 11, 12, 13, 14, 15,  4,  5,  6},
        { 2, 3, 0, 1,   10, 11, 12, 13, 14, 15,  4,  5,  6,  7,  8,  9},
        { 3, 0, 1, 2,   13, 14, 15,  4,  5,  6,  7,  8,  9, 10, 11, 12}};

    int maxvalence = level->getMaxValence();
    int stencilCapacity = 2*maxvalence + 16;
    GregoryBasis::Point P(stencilCapacity), Em(stencilCapacity), Ep(stencilCapacity);

    computeLimitStencils(level, facePoints, extraOrdinaryIndex, fvarChannel, &P, &Em, &Ep);
    P.OffsetIndices(levelVertOffset);
    Em.OffsetIndices(levelVertOffset);
    Ep.OffsetIndices(levelVertOffset);

    // returning patch indices (a mix of cage vertices and patch points)
    int patchPoints[16];

    // first, we traverse the topology to gather 15 vertices. This process is
    // similar to Vtr::Level::gatherQuadRegularInteriorPatchPoints
    int pointIndex = 0;
    int vid = extraOrdinaryIndex;

    // 0-ring
    patchPoints[pointIndex++] = facePoints[0] + levelVertOffset;
    patchPoints[pointIndex++] = facePoints[1] + levelVertOffset;
    patchPoints[pointIndex++] = facePoints[2] + levelVertOffset;
    patchPoints[pointIndex++] = facePoints[3] + levelVertOffset;

    // 1-ring
    ConstIndexArray thisFaceVerts = level->getFaceVertices(thisFace);
    for (int i = 0; i < 4; ++i) {
        Index v = thisFaceVerts[i];
        ConstIndexArray      vFaces   = level->getVertexFaces(v);
        ConstLocalIndexArray vInFaces = level->getVertexFaceLocalIndices(v);

        if (i != vid) {
            // regular corner
            int thisFaceInVFaces = vFaces.FindIndexIn4Tuple(thisFace);

            int intFaceInVFaces  = (thisFaceInVFaces + 2) & 0x3;
            Index intFace    = vFaces[intFaceInVFaces];
            int   vInIntFace = vInFaces[intFaceInVFaces];
            ConstIndexArray facePoints = level->getFaceVertices(intFace);

            patchPoints[pointIndex++] =
                facePoints[(vInIntFace + 1)&3] + levelVertOffset;
            patchPoints[pointIndex++] =
                facePoints[(vInIntFace + 2)&3] + levelVertOffset;
            patchPoints[pointIndex++] =
                facePoints[(vInIntFace + 3)&3] + levelVertOffset;
        } else {
            // irregular corner
            int thisFaceInVFaces = vFaces.FindIndex(thisFace);
            int valence = vFaces.size();
            {
                // first
                int intFaceInVFaces  = (thisFaceInVFaces + 1) % valence;
                Index intFace    = vFaces[intFaceInVFaces];
                int   vInIntFace = vInFaces[intFaceInVFaces];
                ConstIndexArray facePoints = level->getFaceVertices(intFace);
                patchPoints[pointIndex++] =
                    facePoints[(vInIntFace+3)&3] + levelVertOffset;
            }
            {
                // middle: (n-vertices) needs a limit stencil. skip for now
                pointIndex++;
            }
            {
                // end
                int intFaceInVFaces  = (thisFaceInVFaces + (valence-1)) %valence;
                Index intFace    = vFaces[intFaceInVFaces];
                int   vInIntFace = vInFaces[intFaceInVFaces];
                ConstIndexArray facePoints = level->getFaceVertices(intFace);
                patchPoints[pointIndex++] =
                    facePoints[(vInIntFace+1)&3] + levelVertOffset;

            }
        }
    }

    // stencils for patch points
    GregoryBasis::Point X5(stencilCapacity),
        X6(stencilCapacity),
        X7(stencilCapacity),
        X8(stencilCapacity),
        X4(stencilCapacity),
        X15(stencilCapacity),
        X14(stencilCapacity);

    // limit tangent : Em
    // X6 = 1/3 * ( 36Em - 16P0 - 8P1 - 2P2 - 4P3 -  P6 - 2P7)
    // X7 = 1/3 * (-18Em +  8P0 + 4P1 +  P2 + 2P3 + 2P6 + 4P7)
    // X8 = X6 + (P8-P6)
    X6.AddWithWeight(Em,                             36.0f/3.0f);
    X6.AddWithWeight(patchPoints[rotation[vid][0]], -16.0f/3.0f);
    X6.AddWithWeight(patchPoints[rotation[vid][1]],  -8.0f/3.0f);
    X6.AddWithWeight(patchPoints[rotation[vid][2]],  -2.0f/3.0f);
    X6.AddWithWeight(patchPoints[rotation[vid][3]],  -4.0f/3.0f);
    X6.AddWithWeight(patchPoints[rotation[vid][6]],  -1.0f/3.0f);
    X6.AddWithWeight(patchPoints[rotation[vid][7]],  -2.0f/3.0f);

    X7.AddWithWeight(Em,                            -18.0f/3.0f);
    X7.AddWithWeight(patchPoints[rotation[vid][0]],   8.0f/3.0f);
    X7.AddWithWeight(patchPoints[rotation[vid][1]],   4.0f/3.0f);
    X7.AddWithWeight(patchPoints[rotation[vid][2]],   1.0f/3.0f);
    X7.AddWithWeight(patchPoints[rotation[vid][3]],   2.0f/3.0f);
    X7.AddWithWeight(patchPoints[rotation[vid][6]],   2.0f/3.0f);
    X7.AddWithWeight(patchPoints[rotation[vid][7]],   4.0f/3.0f);

    X8 = X6;
    X8.AddWithWeight(patchPoints[rotation[vid][8]], 1.0f);
    X8.AddWithWeight(patchPoints[rotation[vid][6]], -1.0f);

    // limit tangent : Ep
    // X4  = 1/3 * ( 36EP - 16P0 - 4P1 - 2P15 - 2P2 - 8P3 -  P4)
    // X15 = 1/3 * (-18EP +  8P0 + 2P1 + 4P15 +  P2 + 4P3 + 2P4)
    // X14 = X4  + (P14 - P4)
    X4.AddWithWeight(Ep,                             36.0f/3.0f);
    X4.AddWithWeight(patchPoints[rotation[vid][0]], -16.0f/3.0f);
    X4.AddWithWeight(patchPoints[rotation[vid][1]],  -4.0f/3.0f);
    X4.AddWithWeight(patchPoints[rotation[vid][2]],  -2.0f/3.0f);
    X4.AddWithWeight(patchPoints[rotation[vid][3]],  -8.0f/3.0f);
    X4.AddWithWeight(patchPoints[rotation[vid][4]],  -1.0f/3.0f);
    X4.AddWithWeight(patchPoints[rotation[vid][15]], -2.0f/3.0f);

    X15.AddWithWeight(Ep,                            -18.0f/3.0f);
    X15.AddWithWeight(patchPoints[rotation[vid][0]],   8.0f/3.0f);
    X15.AddWithWeight(patchPoints[rotation[vid][1]],   2.0f/3.0f);
    X15.AddWithWeight(patchPoints[rotation[vid][2]],   1.0f/3.0f);
    X15.AddWithWeight(patchPoints[rotation[vid][3]],   4.0f/3.0f);
    X15.AddWithWeight(patchPoints[rotation[vid][4]],   2.0f/3.0f);
    X15.AddWithWeight(patchPoints[rotation[vid][15]],  4.0f/3.0f);

    X14 = X4;
    X14.AddWithWeight(patchPoints[rotation[vid][14]],  1.0f);
    X14.AddWithWeight(patchPoints[rotation[vid][4]],  -1.0f);

    // limit corner (16th free vert)
    // X5 = 36LP - 16P0 - 4(P1 + P3 + P4 + P6) - (P2 + P7 + P15)
    X5.AddWithWeight(P,                              36.0f);
    X5.AddWithWeight(patchPoints[rotation[vid][0]], -16.0f);
    X5.AddWithWeight(patchPoints[rotation[vid][1]],  -4.0f);
    X5.AddWithWeight(patchPoints[rotation[vid][3]],  -4.0f);
    X5.AddWithWeight(X4,                             -4.0f);
    X5.AddWithWeight(X6,                             -4.0f);
    X5.AddWithWeight(patchPoints[rotation[vid][2]],  -1.0f);
    X5.AddWithWeight(X7,                             -1.0f);
    X5.AddWithWeight(X15,                            -1.0f);

    //     [5]   (4)---(15)--(14)    0 : extraordinary vertex
    //            |     |     |
    //            |     |     |      1,2,3,9,10,11,12,13 :
    //     (6)----0-----3-----13       B-Spline control points, gathered by
    //      |     |     |     |         traversing topology
    //      |     |     |     |
    //     (7)----1-----2-----12     (5) :
    //      |     |     |     |        Fitted patch point (from limit position)
    //      |     |     |     |
    //     (8)----9-----10----11     (4),(6),(7),(8),(14),(15) :
    //
    // patch point stencils will be stored in this order
    // (Em) 6, 7, 8, (Ep) 4, 15, 14, (P) 5

    int offset = (fvarChannel < 0)
               ? _refiner->GetNumVerticesTotal()
               : _refiner->GetNumFVarValuesTotal(fvarChannel);

    int varyingIndex0 = facePoints[vid] + levelVertOffset;
    int varyingIndex1 = facePoints[(vid+1)&3] + levelVertOffset;
    int varyingIndex3 = facePoints[(vid+3)&3] + levelVertOffset;

    // push back to stencils;
    patchPoints[3* vid + 6]        = (_numVertices++) + offset;
    GregoryBasis::AppendToStencilTable(X6, _vertexStencils);
    if (_varyingStencils) {
        GregoryBasis::AppendToStencilTable(varyingIndex0, _varyingStencils);
    }

    patchPoints[3*((vid+1)%4) + 4] = (_numVertices++) + offset;
    GregoryBasis::AppendToStencilTable(X7, _vertexStencils);
    if (_varyingStencils) {
        GregoryBasis::AppendToStencilTable(varyingIndex1, _varyingStencils);
    }

    patchPoints[3*((vid+1)%4) + 5] = (_numVertices++) + offset;
    GregoryBasis::AppendToStencilTable(X8, _vertexStencils);
    if (_varyingStencils) {
        GregoryBasis::AppendToStencilTable(varyingIndex1, _varyingStencils);
    }

    patchPoints[3* vid + 4]        = (_numVertices++) + offset;
    GregoryBasis::AppendToStencilTable(X4, _vertexStencils);
    if (_varyingStencils) {
        GregoryBasis::AppendToStencilTable(varyingIndex0, _varyingStencils);
    }

    patchPoints[3*((vid+3)%4) + 6] = (_numVertices++) + offset;
    GregoryBasis::AppendToStencilTable(X15, _vertexStencils);
    if (_varyingStencils) {
        GregoryBasis::AppendToStencilTable(varyingIndex3, _varyingStencils);
    }

    patchPoints[3*((vid+3)%4) + 5] = (_numVertices++) + offset;
    GregoryBasis::AppendToStencilTable(X14, _vertexStencils);
    if (_varyingStencils) {
        GregoryBasis::AppendToStencilTable(varyingIndex3, _varyingStencils);
    }

    patchPoints[3*vid + 5]         = (_numVertices++) + offset;
    GregoryBasis::AppendToStencilTable(X5, _vertexStencils);
    if (_varyingStencils) {
        GregoryBasis::AppendToStencilTable(varyingIndex0, _varyingStencils);
    }

    // reorder into UV row-column
    static int const permuteRegular[16] =
        { 5, 6, 7, 8, 4, 0, 1, 9, 15, 3, 2, 10, 14, 13, 12, 11 };
    for (int i = 0; i < 16; ++i) {
        _patchPoints.push_back(patchPoints[permuteRegular[i]]);
    }
    ++_numPatches;
    return ConstIndexArray(&_patchPoints[(_numPatches-1)*16], 16);
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
