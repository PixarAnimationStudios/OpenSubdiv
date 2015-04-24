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
#ifndef SDC_CATMARK_SCHEME_H
#define SDC_CATMARK_SCHEME_H

#include "../version.h"

#include "../sdc/scheme.h"

#include <cassert>
#include <cmath>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Sdc {

//
//  Specializations for Scheme<SCHEME_CATMARK>:
//

//
//  Catmark traits:
//
template <>
inline Split Scheme<SCHEME_CATMARK>::GetTopologicalSplitType() { return SPLIT_TO_QUADS; }

template <>
inline int Scheme<SCHEME_CATMARK>::GetRegularFaceSize() { return 4; }

template <>
inline int Scheme<SCHEME_CATMARK>::GetRegularVertexValence() { return 4; }

template <>
inline int Scheme<SCHEME_CATMARK>::GetLocalNeighborhoodSize() { return 1; }


//
//  Masks for edge-vertices:  the hard Crease mask does not need to be specialized
//  (simply the midpoint), so all that is left is the Smooth case:
//
//  The Smooth mask is complicated by the need to support the "triangle subdivision"
//  option, which applies different weighting in the presence of triangles.  It is
//  up for debate as to whether this is useful or not -- we may be able to deprecate
//  this option.
//
template <>
template <typename EDGE, typename MASK>
inline void
Scheme<SCHEME_CATMARK>::assignSmoothMaskForEdge(EDGE const& edge, MASK& mask) const {

    typedef typename MASK::Weight Weight;

    int faceCount = edge.GetNumFaces();

    mask.SetNumVertexWeights(2);
    mask.SetNumEdgeWeights(0);
    mask.SetNumFaceWeights(faceCount);
    mask.SetFaceWeightsForFaceCenters(true);

    //
    //  Determine if we need to inspect incident faces and apply alternate weighting for
    //  triangles -- and if so, determine which of the two are triangles.
    //
    bool face0IsTri = false;
    bool face1IsTri = false;
    bool useTriangleOption = (_options.GetTriangleSubdivision() == Options::TRI_SUB_SMOOTH);
    if (useTriangleOption) {
        if (faceCount == 2) {
            //
            //  Ideally we want to avoid this inspection when we have already subdivided at
            //  least once -- need something in the Edge interface to help avoid this, e.g.
            //  an IsRegular() query, the subdivision level...
            //
            int vertsPerFace[2];
            edge.GetNumVerticesPerFace(vertsPerFace);

            face0IsTri = (vertsPerFace[0] == 3);
            face1IsTri = (vertsPerFace[1] == 3);
            useTriangleOption = face0IsTri || face1IsTri;
        } else {
            useTriangleOption = false;
        }
    }

    if (not useTriangleOption) {
        mask.VertexWeight(0) = 0.25f;
        mask.VertexWeight(1) = 0.25f;

        if (faceCount == 2) {
            mask.FaceWeight(0) = 0.25f;
            mask.FaceWeight(1) = 0.25f;
        } else {
            Weight fWeight = 0.5f / (Weight)faceCount;
            for (int i = 0; i < faceCount; ++i) {
                mask.FaceWeight(i) = fWeight;
            }
        }
    } else {
        //
        //  This mimics the implementation in Hbr in terms of order of operations.
        //
        const Weight CATMARK_SMOOTH_TRI_EDGE_WEIGHT = 0.470f;

        Weight f0Weight = face0IsTri ? CATMARK_SMOOTH_TRI_EDGE_WEIGHT : 0.25f;
        Weight f1Weight = face1IsTri ? CATMARK_SMOOTH_TRI_EDGE_WEIGHT : 0.25f;

        Weight fWeight = 0.5f * (f0Weight + f1Weight);
        Weight vWeight = 0.5f * (1.0f - 2.0f * fWeight);

        mask.VertexWeight(0) = vWeight;
        mask.VertexWeight(1) = vWeight;

        mask.FaceWeight(0) = fWeight;
        mask.FaceWeight(1) = fWeight;
    }
}


//
//  Masks for vertex-vertices:  the hard Corner mask does not need to be specialized
//  (simply the vertex itself), leaving the Crease and Smooth cases (Dart is smooth):
//
template <>
template <typename VERTEX, typename MASK>
inline void
Scheme<SCHEME_CATMARK>::assignCreaseMaskForVertex(VERTEX const& vertex, MASK& mask,
                                                  int const creaseEnds[2]) const {
    typedef typename MASK::Weight Weight;

    int valence = vertex.GetNumEdges();

    mask.SetNumVertexWeights(1);
    mask.SetNumEdgeWeights(valence);
    mask.SetNumFaceWeights(0);
    mask.SetFaceWeightsForFaceCenters(false);

    Weight vWeight = 0.75f;
    Weight eWeight = 0.125f;

    mask.VertexWeight(0) = vWeight;
    for (int i = 0; i < valence; ++i) {
        mask.EdgeWeight(i) = 0.0f;
    }
    mask.EdgeWeight(creaseEnds[0]) = eWeight;
    mask.EdgeWeight(creaseEnds[1]) = eWeight;
}

template <>
template <typename VERTEX, typename MASK>
inline void
Scheme<SCHEME_CATMARK>::assignSmoothMaskForVertex(VERTEX const& vertex, MASK& mask) const {

    typedef typename MASK::Weight Weight;

    //
    //  Remember that when the edge- and face-counts differ, we need to adjust this...
    //
    //  Keep what's below for eCount == fCount and for the other cases -- which should
    //  only occur for non-manifold vertices -- use the following formula that we've
    //  adapted in MM:
    //
    //      v' = (F + 2*E + (n-3)*v) / n
    //
    //  where F is the average of the face points (fi) and E is the average of the edge
    //  midpoints (ei).  The F term gives is the 1/(n*n) of below and we just need to
    //  factor the E and v terms to account for the edge endpoints rather than midpoints.
    //
    int valence = vertex.GetNumFaces();

    mask.SetNumVertexWeights(1);
    mask.SetNumEdgeWeights(valence);
    mask.SetNumFaceWeights(valence);
    mask.SetFaceWeightsForFaceCenters(true);

    Weight vWeight = (Weight)(valence - 2) / (Weight)valence;
    Weight fWeight = 1.0f / (Weight)(valence * valence);
    Weight eWeight = fWeight;

    mask.VertexWeight(0) = vWeight;
    for (int i = 0; i < valence; ++i) {
        mask.EdgeWeight(i) = eWeight;
        mask.FaceWeight(i) = fWeight;
    }
}

//
//  Limit masks for position:
//
template <>
template <typename VERTEX, typename MASK>
inline void
Scheme<SCHEME_CATMARK>::assignCornerLimitMask(VERTEX const& /* vertex */, MASK& posMask) const {

    posMask.SetNumVertexWeights(1);
    posMask.SetNumEdgeWeights(0);
    posMask.SetNumFaceWeights(0);
    posMask.SetFaceWeightsForFaceCenters(false);

    posMask.VertexWeight(0) = 1.0f;
}

template <>
template <typename VERTEX, typename MASK>
inline void
Scheme<SCHEME_CATMARK>::assignCreaseLimitMask(VERTEX const& vertex, MASK& posMask,
                                              int const creaseEnds[2]) const {

    typedef typename MASK::Weight Weight;

    int valence = vertex.GetNumEdges();

    posMask.SetNumVertexWeights(1);
    posMask.SetNumEdgeWeights(valence);
    posMask.SetNumFaceWeights(0);
    posMask.SetFaceWeightsForFaceCenters(false);

    Weight vWeight = 2.0f / 3.0f;
    Weight eWeight = 1.0f / 6.0f;

    posMask.VertexWeight(0) = vWeight;
    for (int i = 0; i < valence; ++i) {
        posMask.EdgeWeight(i) = 0.0f;
    }
    posMask.EdgeWeight(creaseEnds[0]) = eWeight;
    posMask.EdgeWeight(creaseEnds[1]) = eWeight;
}

template <>
template <typename VERTEX, typename MASK>
inline void
Scheme<SCHEME_CATMARK>::assignSmoothLimitMask(VERTEX const& vertex, MASK& posMask) const {

    typedef typename MASK::Weight Weight;

    int valence = vertex.GetNumFaces();
    assert(valence != 2);

    posMask.SetNumVertexWeights(1);
    posMask.SetNumEdgeWeights(valence);
    posMask.SetNumFaceWeights(valence);
    posMask.SetFaceWeightsForFaceCenters(false);

    //  Specialize for the regular case:
    Weight fWeight = 1.0f / 36.0f;
    Weight eWeight = 1.0f /  9.0f;
    Weight vWeight = 4.0f /  9.0f;

    if (valence != 4) {
        fWeight = 1.0f / (Weight)(valence * (valence + 5.0f));
        eWeight = 4.0f * fWeight;
        vWeight = (Weight)(1.0f - valence * (eWeight + fWeight));
    }

    posMask.VertexWeight(0) = vWeight;
    for (int i = 0; i < valence; ++i) {
        posMask.EdgeWeight(i) = eWeight;
        posMask.FaceWeight(i) = fWeight;
    }
}

//
//  Limit masks for tangents -- these are stubs for now, or have a temporary
//  implementation
//
template <>
template <typename VERTEX, typename MASK>
inline void
Scheme<SCHEME_CATMARK>::assignCornerLimitTangentMasks(VERTEX const& vertex,
        MASK& tan1Mask, MASK& tan2Mask) const {

    int valence = vertex.GetNumEdges();

    tan1Mask.SetNumVertexWeights(1);
    tan1Mask.SetNumEdgeWeights(valence);
    tan1Mask.SetNumFaceWeights(0);
    tan1Mask.SetFaceWeightsForFaceCenters(false);

    tan2Mask.SetNumVertexWeights(1);
    tan2Mask.SetNumEdgeWeights(valence);
    tan2Mask.SetNumFaceWeights(0);
    tan2Mask.SetFaceWeightsForFaceCenters(false);

    //  Should be at least 2 edges -- be sure to clear weights for any more:
    tan1Mask.VertexWeight(0) = -1.0f;
    tan1Mask.EdgeWeight(0)   =  1.0f;
    tan1Mask.EdgeWeight(1)   =  0.0f;

    tan2Mask.VertexWeight(0) = -1.0f;
    tan2Mask.EdgeWeight(0)   =  0.0f;
    tan2Mask.EdgeWeight(1)   =  1.0f;

    for (int i = 2; i < valence; ++i) {
        tan1Mask.EdgeWeight(i) = 0.0f;
        tan2Mask.EdgeWeight(i) = 0.0f;
    }
}

template <>
template <typename VERTEX, typename MASK>
inline void
Scheme<SCHEME_CATMARK>::assignCreaseLimitTangentMasks(VERTEX const& vertex,
        MASK& tan1Mask, MASK& tan2Mask, int const creaseEnds[2]) const {

    int valence = vertex.GetNumEdges();

    tan1Mask.SetNumVertexWeights(1);
    tan1Mask.SetNumEdgeWeights(valence);
    tan1Mask.SetNumFaceWeights(0);
    tan1Mask.SetFaceWeightsForFaceCenters(false);

    tan2Mask.SetNumVertexWeights(1);
    tan2Mask.SetNumEdgeWeights(valence);
    tan2Mask.SetNumFaceWeights(0);
    tan2Mask.SetFaceWeightsForFaceCenters(false);

    //  Specialize for the regular (boundary) case:
    bool isRegular = (vertex.GetNumEdges() == 3);
    if (isRegular) {
        tan1Mask.VertexWeight(0) = 0.0f;
        tan1Mask.EdgeWeight(0) =  1.0f;
        tan1Mask.EdgeWeight(1) =  0.0f;
        tan1Mask.EdgeWeight(2) = -1.0f;

        tan2Mask.VertexWeight(0) = -1.0f;
        tan2Mask.EdgeWeight(0) =  0.0f;
        tan2Mask.EdgeWeight(1) =  1.0f;
        tan2Mask.EdgeWeight(2) =  0.0f;
    } else {
        //  First, the tangent along the crease:
        tan1Mask.VertexWeight(0) = 0.0f;
        for (int i = 0; i < valence; ++i) {
            tan1Mask.EdgeWeight(i) = 0.0f;
        }
        tan1Mask.EdgeWeight(creaseEnds[0]) =  1.0f;
        tan1Mask.EdgeWeight(creaseEnds[1]) = -1.0f;

        //  Second, the tangent across the interior faces:
        //      - just using an interior edge for now
        //      - ultimately need regular and extra-ordinary cases here:
        //
        tan2Mask.VertexWeight(0) = -1.0f;
        for (int i = 0; i < valence; ++i) {
            tan2Mask.EdgeWeight(i) = 0.0f;
        }
        tan2Mask.EdgeWeight((creaseEnds[0] + creaseEnds[1]) >> 1) =  1.0f;
    }
}

template <>
template <typename VERTEX, typename MASK>
inline void
Scheme<SCHEME_CATMARK>::assignSmoothLimitTangentMasks(VERTEX const& vertex,
        MASK& tan1Mask, MASK& tan2Mask) const {

    typedef typename MASK::Weight Weight;

    int valence = vertex.GetNumFaces();
    assert(valence != 2);

    //  Using the Loop tangent masks for now...
    tan1Mask.SetNumVertexWeights(1);
    tan1Mask.SetNumEdgeWeights(valence);
    tan1Mask.SetNumFaceWeights(0);
    tan1Mask.SetFaceWeightsForFaceCenters(false);

    tan2Mask.SetNumVertexWeights(1);
    tan2Mask.SetNumEdgeWeights(valence);
    tan2Mask.SetNumFaceWeights(0);
    tan2Mask.SetFaceWeightsForFaceCenters(false);

    tan1Mask.VertexWeight(0) = 0.0f;
    tan2Mask.VertexWeight(0) = 0.0f;

    Weight alpha = (Weight) (2.0f * M_PI / valence);
    for (int i = 0; i < valence; ++i) {
        double alphaI = alpha * i;
        tan1Mask.EdgeWeight(i) = std::cos(alphaI);
        tan2Mask.EdgeWeight(i) = std::sin(alphaI);
    }
}

} // end namespace sdc

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;
} // end namespace OpenSubdiv

#endif /* SDC_CATMARK_SCHEME_H */
