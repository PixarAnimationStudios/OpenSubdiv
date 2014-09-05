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
//  Specializations for Scheme<TYPE_CATMARK>:
//

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
Scheme<TYPE_CATMARK>::assignSmoothMaskForEdge(EDGE const& edge, MASK& mask) const {

    typedef typename MASK::Weight Weight;

    int faceCount = edge.GetNumFaces();

    mask.SetNumVertexWeights(2);
    mask.SetNumEdgeWeights(0);
    mask.SetNumFaceWeights(faceCount);

    //
    //  Determine if we need to inspect incident faces and apply alternate weighting for
    //  triangles -- and if so, determine which of the two are triangles.
    //
    //  (Is this really used?  Would be nice if we could deprecate this option...)
    //
    bool face0IsTri = false;
    bool face1IsTri = false;
    bool useTriangleOption = (_options.GetTriangleSubdivision() != Options::TRI_SUB_NORMAL);
    if (useTriangleOption) {
        if (faceCount == 2) {
            //
            //  Need to inspect/gather valence of incident faces here...
            //
            useTriangleOption = face0IsTri || face1IsTri;
        } else {
            useTriangleOption = false;
        }
    }

    if (!useTriangleOption) {
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
        //  This mimics the implementation in Hbr in terms of order of operations.  If
        //  the triangle-subdivision option can be deprecated we can remove this block:
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
Scheme<TYPE_CATMARK>::assignCreaseMaskForVertex(VERTEX const& vertex, MASK& mask, float const edgeSharpness[]) const {

    typedef typename MASK::Weight Weight;

    int valence = vertex.GetNumEdges();

    mask.SetNumVertexWeights(1);
    mask.SetNumEdgeWeights(valence);
    mask.SetNumFaceWeights(0);

    Weight vWeight = 0.75f;
    Weight eWeight = 0.125f;

    mask.VertexWeight(0) = vWeight;

    //
    //  NOTE -- at some point the sharpness vector was optional, and topology would be used
    //  to identify a boundary crease.  We are currently no longer passing a null sharpness
    //  vector and may not support it in future, in which case this test can be removed:
    //
    if (edgeSharpness != 0) {
        //  Use the sharpness values to identify the crease edges:
        for (int i = 0; i < valence; ++i) {
            mask.EdgeWeight(i) = (edgeSharpness[i] > 0.0f) ? eWeight : 0.0f;
        }
    } else {
        //  Use the boundary edges (first and last) as the crease edges:
        mask.EdgeWeight(0) = eWeight;
        for (int i = 1; i < (valence - 1); ++i) {
            mask.EdgeWeight(i) = 0.0f;
        }
        mask.EdgeWeight(valence-1) = eWeight;
    }
}

template <>
template <typename VERTEX, typename MASK>
inline void
Scheme<TYPE_CATMARK>::assignSmoothMaskForVertex(VERTEX const& vertex, MASK& mask) const {

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
Scheme<TYPE_CATMARK>::assignBoundaryLimitMask(VERTEX const& vertex, MASK& posMask) const {

    typedef typename MASK::Weight Weight;

    int valence = vertex.GetNumEdges();

    posMask.SetNumVertexWeights(1);
    posMask.SetNumEdgeWeights(valence);
    posMask.SetNumFaceWeights(0);

    Weight vWeight = 2.0f / 3.0f;
    Weight eWeight = 1.0f / 6.0f;

    posMask.VertexWeight(0) = vWeight;
    posMask.EdgeWeight(0) = eWeight;
    for (int i = 1; i < valence - 1; ++i) {
        posMask.EdgeWeight(i) = 0.0f;
    }
    posMask.EdgeWeight(valence - 1) = eWeight;
}

template <>
template <typename VERTEX, typename MASK>
inline void
Scheme<TYPE_CATMARK>::assignInteriorLimitMask(VERTEX const& vertex, MASK& posMask) const {

    typedef typename MASK::Weight Weight;

    int valence = vertex.GetNumFaces();
    assert(valence != 2);

    posMask.SetNumVertexWeights(1);
    posMask.SetNumEdgeWeights(valence);
    posMask.SetNumFaceWeights(valence);

    //  Probably a good idea to test for and assign the regular case as a special case:

    Weight fWeight = 1.0f / (Weight)(valence * (valence + 5.0f));
    Weight eWeight = 4.0f * fWeight;
    Weight vWeight = (Weight)(1.0f - valence * (eWeight + fWeight));

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
Scheme<TYPE_CATMARK>::assignBoundaryLimitTangentMasks(VERTEX const& vertex,
        MASK& tan1Mask, MASK& tan2Mask) const {

    tan1Mask.SetNumVertexWeights(1);
    tan1Mask.SetNumEdgeWeights(0);
    tan1Mask.SetNumFaceWeights(0);
    tan1Mask.VertexWeight(0) = 0.0f;

    tan2Mask.SetNumVertexWeights(1);
    tan2Mask.SetNumEdgeWeights(0);
    tan2Mask.SetNumFaceWeights(0);
    tan2Mask.VertexWeight(0) = 0.0f;
}

template <>
template <typename VERTEX, typename MASK>
inline void
Scheme<TYPE_CATMARK>::assignInteriorLimitTangentMasks(VERTEX const& vertex,
        MASK& tan1Mask, MASK& tan2Mask) const {

    typedef typename MASK::Weight Weight;

    int valence = vertex.GetNumFaces();
    assert(valence != 2);

    //  Using the Loop tangent masks for now...
    tan1Mask.SetNumVertexWeights(1);
    tan1Mask.SetNumEdgeWeights(valence);
    tan1Mask.SetNumFaceWeights(0);

    tan2Mask.SetNumVertexWeights(1);
    tan2Mask.SetNumEdgeWeights(valence);
    tan2Mask.SetNumFaceWeights(0);

    tan1Mask.VertexWeight(0) = 0.0f;
    tan2Mask.VertexWeight(0) = 0.0f;

    Weight alpha = (Weight) (2.0f * M_PI / valence);
    for (int i = 0; i < valence; ++i) {
        double alphaI = alpha * i;
        tan1Mask.EdgeWeight(i) = cos(alphaI);
        tan2Mask.EdgeWeight(i) = sin(alphaI);
    }
}

} // end namespace sdc

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;
} // end namespace OpenSubdiv

#endif /* SDC_CATMARK_SCHEME_H */
