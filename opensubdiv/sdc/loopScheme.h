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
#ifndef SDC_LOOP_SCHEME_H
#define SDC_LOOP_SCHEME_H

#include "../version.h"

#include "../sdc/scheme.h"

#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {
namespace Sdc {


//
//  Specializations for Sdc::Scheme<SCHEME_LOOP>:
//
//

//
//  Loop traits:
//
template <>
inline Split Scheme<SCHEME_LOOP>::GetTopologicalSplitType() { return SPLIT_TO_TRIS; }

template <>
inline int Scheme<SCHEME_LOOP>::GetRegularFaceSize() { return 3; }

template <>
inline int Scheme<SCHEME_LOOP>::GetRegularVertexValence() { return 6; }

template <>
inline int Scheme<SCHEME_LOOP>::GetLocalNeighborhoodSize() { return 1; }


//
//  Protected methods to assign the two types of masks for an edge-vertex --
//  Crease and Smooth.
//
//  The Crease case does not really need to be speciailized, though it may be
//  preferable to define all explicitly here.
//
template <>
template <typename EDGE, typename MASK>
inline void
Scheme<SCHEME_LOOP>::assignCreaseMaskForEdge(EDGE const&, MASK& mask) const
{
    mask.SetNumVertexWeights(2);
    mask.SetNumEdgeWeights(0);
    mask.SetNumFaceWeights(0);
    mask.SetFaceWeightsForFaceCenters(false);

    mask.VertexWeight(0) = 0.5f;
    mask.VertexWeight(1) = 0.5f;
}

template <>
template <typename EDGE, typename MASK>
inline void
Scheme<SCHEME_LOOP>::assignSmoothMaskForEdge(EDGE const& edge, MASK& mask) const
{
    int faceCount = edge.GetNumFaces();

    mask.SetNumVertexWeights(2);
    mask.SetNumEdgeWeights(0);
    mask.SetNumFaceWeights(faceCount);
    mask.SetFaceWeightsForFaceCenters(false);

    //
    //  This is where we run into the issue of "face weights" -- we want to weight the
    //  face-centers for Catmark, but face-centers are not generated for Loop.  So do
    //  we make assumptions on how the mask is used, assign some property to the mask
    //  to indicate how they were assigned, or take input from the mask itself?
    //
    //  Regardless, we have two choices:
    //      - face-weights are for the vertices opposite the edge (as in Hbr):
    //          vertex weights = 0.375f;
    //          face weights   = 0.125f;
    //
    //      - face-weights are for the face centers:
    //          vertex weights = 0.125f;
    //          face weights   = 0.375f;
    //
    //  Coincidentally the coefficients are the same but reversed.
    //
    typedef typename MASK::Weight Weight;

    Weight vWeight = mask.AreFaceWeightsForFaceCenters() ? 0.125f : 0.375f;
    Weight fWeight = mask.AreFaceWeightsForFaceCenters() ? 0.375f : 0.125f;

    mask.VertexWeight(0) = vWeight;
    mask.VertexWeight(1) = vWeight;

    if (faceCount == 2) {
        mask.FaceWeight(0) = fWeight;
        mask.FaceWeight(1) = fWeight;
    } else {
        //  The non-manifold case is not clearly defined -- we adjust the above
        //  face-weight to preserve the ratio of edge-center and face-centers:
        fWeight *= 2.0f / (Weight) faceCount;
        for (int i = 0; i < faceCount; ++i) {
            mask.FaceWeight(i) = fWeight;
        }
    }
}


//
//  Protected methods to assign the three types of masks for a vertex-vertex --
//  Corner, Crease and Smooth (Dart is the same as Smooth).
//
//  Corner and Crease do not really need to be speciailized, though it may be
//  preferable to define all explicitly here.
//
template <>
template <typename VERTEX, typename MASK>
inline void
Scheme<SCHEME_LOOP>::assignCornerMaskForVertex(VERTEX const&, MASK& mask) const
{
    mask.SetNumVertexWeights(1);
    mask.SetNumEdgeWeights(0);
    mask.SetNumFaceWeights(0);
    mask.SetFaceWeightsForFaceCenters(false);

    mask.VertexWeight(0) = 1.0f;
}

template <>
template <typename VERTEX, typename MASK>
inline void
Scheme<SCHEME_LOOP>::assignCreaseMaskForVertex(VERTEX const& vertex, MASK& mask, float const edgeSharpness[]) const
{
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
        mask.EdgeWeight(i) = (edgeSharpness[i] > 0.0f) ? eWeight : 0.0f;
    }
}

template <>
template <typename VERTEX, typename MASK>
inline void
Scheme<SCHEME_LOOP>::assignSmoothMaskForVertex(VERTEX const& vertex, MASK& mask) const
{
    typedef typename MASK::Weight Weight;

    int valence = vertex.GetNumFaces();

    mask.SetNumVertexWeights(1);
    mask.SetNumEdgeWeights(valence);
    mask.SetNumFaceWeights(0);
    mask.SetFaceWeightsForFaceCenters(false);

    //  Specialize for the regular case:  1/16 per edge-vert, 5/8 for the vert itself:
    Weight eWeight = (Weight) 0.0625f;
    Weight vWeight = (Weight) 0.625f;

    if (valence != 6) {
        //  From HbrLoopSubdivision<T>::Subdivide(mesh, vertex):
        //     - could use some lookup tables here for common irregular valence (5, 7, 8)
        //       or all of these cosf() calls will be adding up...

        Weight invValence = 1.0f / (Weight) valence;
        Weight beta       = 0.25f * cosf((Weight)M_PI * 2.0f * invValence) + 0.375f;

        eWeight = (0.625f - (beta * beta)) * invValence;;
        vWeight = 1.0f - (eWeight * (Weight)valence);
    }

    mask.VertexWeight(0) = vWeight;
    for (int i = 0; i < valence; ++i) {
        mask.EdgeWeight(i) = eWeight;
    }
}


//
//  Limit masks for position:
//
template <>
template <typename VERTEX, typename MASK>
inline void
Scheme<SCHEME_LOOP>::assignBoundaryLimitMask(VERTEX const& vertex, MASK& posMask) const {

    typedef typename MASK::Weight Weight;

    int valence = vertex.GetNumEdges();

    posMask.SetNumVertexWeights(1);
    posMask.SetNumEdgeWeights(valence);
    posMask.SetNumFaceWeights(0);
    posMask.SetFaceWeightsForFaceCenters(false);

    Weight vWeight = 4.0f / 6.0f;
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
Scheme<SCHEME_LOOP>::assignInteriorLimitMask(VERTEX const& vertex, MASK& posMask) const {

    typedef typename MASK::Weight Weight;

    int valence = vertex.GetNumFaces();
    assert(valence != 2);

    posMask.SetNumVertexWeights(1);
    posMask.SetNumEdgeWeights(valence);
    posMask.SetNumFaceWeights(0);
    posMask.SetFaceWeightsForFaceCenters(false);

    //  Specialize for the regular case:  1/12 per edge-vert, 1/2 for the vert itself:
    Weight eWeight = 1.0f / 12.0f;
    Weight vWeight = 0.5f;

    if (valence != 6) {
        Weight invValence = 1.0f / valence;

        Weight beta = 0.25f * cosf((Weight)M_PI * 2.0f * invValence) + 0.375f;
        beta = (0.625f - (beta * beta)) * invValence;;

        eWeight = 1.0f / (valence + 3.0f / (8.0f * beta));
        vWeight = (Weight)(1.0f - (eWeight * valence));
    }

    posMask.VertexWeight(0) = vWeight;
    for (int i = 0; i < valence; ++i) {
        posMask.EdgeWeight(i) = eWeight;
    }
}

//
//  Limit masks for tangents:
//
template <>
template <typename VERTEX, typename MASK>
inline void
Scheme<SCHEME_LOOP>::assignBoundaryLimitTangentMasks(VERTEX const& /* vertex */,
        MASK& tan1Mask, MASK& tan2Mask) const {

    //  Need to dig up formulae for this case...

    tan1Mask.SetNumVertexWeights(1);
    tan1Mask.SetNumEdgeWeights(0);
    tan1Mask.SetNumFaceWeights(0);
    tan1Mask.SetFaceWeightsForFaceCenters(false);
    tan1Mask.VertexWeight(0) = 0.0f;

    tan2Mask.SetNumVertexWeights(1);
    tan2Mask.SetNumEdgeWeights(0);
    tan2Mask.SetNumFaceWeights(0);
    tan2Mask.SetFaceWeightsForFaceCenters(false);
    tan2Mask.VertexWeight(0) = 0.0f;
}

template <>
template <typename VERTEX, typename MASK>
inline void
Scheme<SCHEME_LOOP>::assignInteriorLimitTangentMasks(VERTEX const& vertex,
        MASK& tan1Mask, MASK& tan2Mask) const {

    typedef typename MASK::Weight Weight;

    int valence = vertex.GetNumFaces();
    assert(valence != 2);

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
        tan1Mask.EdgeWeight(i) = cos(alphaI);
        tan2Mask.EdgeWeight(i) = sin(alphaI);
    }
}

} // end namespace Sdc
} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;
} // end namespace OpenSubdiv

#endif /* SDC_LOOP_SCHEME_H */
