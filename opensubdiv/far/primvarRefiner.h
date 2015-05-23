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
#ifndef OPENSUBDIV3_FAR_PRIMVAR_REFINER_H
#define OPENSUBDIV3_FAR_PRIMVAR_REFINER_H

#include "../version.h"

#include "../sdc/types.h"
#include "../sdc/options.h"
#include "../sdc/bilinearScheme.h"
#include "../sdc/catmarkScheme.h"
#include "../sdc/loopScheme.h"
#include "../vtr/level.h"
#include "../vtr/fvarLevel.h"
#include "../vtr/refinement.h"
#include "../vtr/fvarRefinement.h"
#include "../vtr/stackBuffer.h"
#include "../vtr/maskInterfaces.h"
#include "../far/types.h"
#include "../far/error.h"
#include "../far/topologyLevel.h"
#include "../far/topologyRefiner.h"

#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Vtr { class SparseSelector; }

namespace Far {


///
///  \brief Applies refinement operations to generic primvar data.
///
class PrimvarRefiner {

public:
    PrimvarRefiner(TopologyRefiner const & refiner) : _refiner(refiner) { }
    ~PrimvarRefiner() { }

    TopologyRefiner const & GetTopologyRefiner() const { return _refiner; }

    //@{
    ///  @name Primvar data interpolation
    ///
    /// \anchor templating
    ///
    /// \note Interpolation methods template both the source and destination
    ///       data buffer classes. Client-code is expected to provide interfaces
    ///       that implement the functions specific to its primitive variable
    ///       data layout. Template APIs must implement the following:
    ///       <br><br> \code{.cpp}
    ///
    ///       class MySource {
    ///           MySource & operator[](int index);
    ///       };
    ///
    ///       class MyDestination {
    ///           void Clear();
    ///           void AddWithWeight(MySource const & value, float weight);
    ///           void AddWithWeight(MyDestination const & value, float weight);
    ///
    ///           // optional
    ///           void AddVaryingWithWeight(MySource const & value, float weight);
    ///       };
    ///
    ///       \endcode
    ///       <br>
    ///       It is possible to implement a single interface only and use it as
    ///       both source and destination.
    ///       <br><br>
    ///       Primitive variable buffers are expected to be arrays of instances,
    ///       passed either as direct pointers or with a container
    ///       (ex. std::vector<MyVertex>).
    ///       Some interpolation methods however allow passing the buffers by
    ///       reference: this allows to work transparently with arrays and
    ///       containers (or other scheme that overload the '[]' operator)
    ///       <br><br>
    ///       See the <a href=http://graphics.pixar.com/opensubdiv/docs/tutorials.html>
    ///       Far tutorials</a> for code examples.
    ///

    /// \brief Apply vertex and varying interpolation weights to a primvar
    ///        buffer
    ///
    /// The destination buffer must allocate an array of data for all the refined
    /// vertices (at least GetNumVerticesTotal()-GetLevel(0).GetNumVertices())
    ///
    /// @param src  Source primvar buffer (\ref templating control vertex data)
    ///
    /// @param dst  Destination primvar buffer (\ref templating refined vertex data)
    ///
    template <class T, class U> void Interpolate(T const * src, U * dst) const;

    /// \brief Apply vertex and varying interpolation weights to a primvar
    ///        buffer for a single level
    /// level of refinement.
    ///
    /// The destination buffer must allocate an array of data for all the
    /// refined vertices (at least GetLevel(level).GetNumVertices())
    ///
    /// @param level  The refinement level
    ///
    /// @param src    Source primvar buffer (\ref templating control vertex data)
    ///
    /// @param dst    Destination primvar buffer (\ref templating refined vertex data)
    ///
    template <class T, class U> void Interpolate(int level, T const & src, U & dst) const;


    /// \brief Apply only varying interpolation weights to a primvar buffer
    ///
    /// This method can be a useful alternative if the varying primvar data
    /// does not need to be re-computed over time.
    ///
    /// The destination buffer must allocate an array of data for all the refined
    /// vertices (at least GetNumVerticesTotal()-GetLevel(0).GetNumVertices())
    ///
    /// @param src  Source primvar buffer (\ref templating control vertex data)
    ///
    /// @param dst  Destination primvar buffer (\ref templating refined vertex data)
    ///
    template <class T, class U> void InterpolateVarying(T const * src, U * dst) const;

    /// \brief Apply only varying interpolation weights to a primvar buffer
    ///        for a single level level of refinement.
    ///
    /// This method can be a useful alternative if the varying primvar data
    /// does not need to be re-computed over time.
    ///
    /// The destination buffer must allocate an array of data for all the
    /// refined vertices (at least GetLevel(level).GetNumVertices())
    ///
    /// @param level  The refinement level
    ///
    /// @param src    Source primvar buffer (\ref templating control vertex data)
    ///
    /// @param dst    Destination primvar buffer (\ref templating refined vertex data)
    ///
    template <class T, class U> void InterpolateVarying(int level, T const & src, U & dst) const;

    /// \brief Apply face-varying interpolation weights to a primvar buffer
    //         associated with a particular face-varying channel
    ///
    template <class T, class U> void InterpolateFaceVarying(T const * src, U * dst, int channel = 0) const;

    template <class T, class U> void InterpolateFaceVarying(int level, T const & src, U & dst, int channel = 0) const;

    template <class T, class U> void LimitFaceVarying(T const & src, U * dst, int channel = 0) const;


    /// \brief Apply vertex interpolation limit weights to a primvar buffer
    ///
    /// The source buffer must refer to an array of previously interpolated
    /// vertex data for the last refinement level.  The destination buffer
    /// must allocate an array for all vertices at the last refinement level
    /// (at least GetLevel(GetMaxLevel()).GetNumVertices())
    ///
    /// @param src  Source primvar buffer (refined vertex data) for last level
    ///
    /// @param dst  Destination primvar buffer (vertex data at the limit)
    ///
    template <class T, class U> void Limit(T const & src, U & dstPos) const;

    template <class T, class U, class U1, class U2>
    void Limit(T const & src, U & dstPos, U1 & dstTan1, U2 & dstTan2) const;

    //@}

private:

    //  Non-copyable:
    PrimvarRefiner(PrimvarRefiner const & src) : _refiner(src._refiner) { }
    PrimvarRefiner & operator=(PrimvarRefiner const &) { return *this; }

    template <Sdc::SchemeType SCHEME, class T, class U> void interpolateChildVertsFromFaces(Vtr::Refinement const &, T const & src, U & dst) const;
    template <Sdc::SchemeType SCHEME, class T, class U> void interpolateChildVertsFromEdges(Vtr::Refinement const &, T const & src, U & dst) const;
    template <Sdc::SchemeType SCHEME, class T, class U> void interpolateChildVertsFromVerts(Vtr::Refinement const &, T const & src, U & dst) const;

    template <class T, class U> void varyingInterpolateChildVertsFromFaces(Vtr::Refinement const &, T const & src, U & dst) const;
    template <class T, class U> void varyingInterpolateChildVertsFromEdges(Vtr::Refinement const &, T const & src, U & dst) const;
    template <class T, class U> void varyingInterpolateChildVertsFromVerts(Vtr::Refinement const &, T const & src, U & dst) const;

    template <Sdc::SchemeType SCHEME, class T, class U> void faceVaryingInterpolateChildVertsFromFaces(Vtr::Refinement const &, T const & src, U & dst, int channel) const;
    template <Sdc::SchemeType SCHEME, class T, class U> void faceVaryingInterpolateChildVertsFromEdges(Vtr::Refinement const &, T const & src, U & dst, int channel) const;
    template <Sdc::SchemeType SCHEME, class T, class U> void faceVaryingInterpolateChildVertsFromVerts(Vtr::Refinement const &, T const & src, U & dst, int channel) const;

    template <Sdc::SchemeType SCHEME, class T, class U, class U1, class U2> void limit(T const & src, U & pos, U1 * tan1, U2 * tan2) const;

    template <Sdc::SchemeType SCHEME, class T, class U> void faceVaryingLimit(T const & src, U * dst, int channel) const;

private:

    TopologyRefiner const &  _refiner;
};



template <class T, class U>
inline void
PrimvarRefiner::Interpolate(T const * src, U * dst) const {

    for (int level = 1; level <= _refiner.GetMaxLevel(); ++level) {

        Interpolate(level, src, dst);

        src = dst;
        dst += _refiner.GetLevel(level).GetNumVertices();
    }
}

template <class T, class U>
inline void
PrimvarRefiner::Interpolate(int level, T const & src, U & dst) const {

    assert(level>0 and level<=(int)_refiner._refinements.size());

    Vtr::Refinement const & refinement = _refiner.getRefinement(level-1);

    switch (_refiner._subdivType) {
    case Sdc::SCHEME_CATMARK:
        interpolateChildVertsFromFaces<Sdc::SCHEME_CATMARK>(refinement, src, dst);
        interpolateChildVertsFromEdges<Sdc::SCHEME_CATMARK>(refinement, src, dst);
        interpolateChildVertsFromVerts<Sdc::SCHEME_CATMARK>(refinement, src, dst);
        break;
    case Sdc::SCHEME_LOOP:
        interpolateChildVertsFromFaces<Sdc::SCHEME_LOOP>(refinement, src, dst);
        interpolateChildVertsFromEdges<Sdc::SCHEME_LOOP>(refinement, src, dst);
        interpolateChildVertsFromVerts<Sdc::SCHEME_LOOP>(refinement, src, dst);
        break;
    case Sdc::SCHEME_BILINEAR:
        interpolateChildVertsFromFaces<Sdc::SCHEME_BILINEAR>(refinement, src, dst);
        interpolateChildVertsFromEdges<Sdc::SCHEME_BILINEAR>(refinement, src, dst);
        interpolateChildVertsFromVerts<Sdc::SCHEME_BILINEAR>(refinement, src, dst);
        break;
    }
}

template <Sdc::SchemeType SCHEME, class T, class U>
inline void
PrimvarRefiner::interpolateChildVertsFromFaces(
    Vtr::Refinement const & refinement, T const & src, U & dst) const {

    if (refinement.getNumChildVerticesFromFaces() == 0) return;

    Sdc::Scheme<SCHEME> scheme(_refiner._subdivOptions);

    const Vtr::Level& parent = refinement.parent();

    Vtr::internal::StackBuffer<float,16> fVertWeights(parent.getMaxValence());

    for (int face = 0; face < parent.getNumFaces(); ++face) {

        Vtr::Index cVert = refinement.getFaceChildVertex(face);
        if (!Vtr::IndexIsValid(cVert))
            continue;

        //  Declare and compute mask weights for this vertex relative to its parent face:
        ConstIndexArray fVerts = parent.getFaceVertices(face);

        float fVaryingWeight = 1.0f / (float) fVerts.size();

        Vtr::MaskInterface fMask(fVertWeights, 0, 0);
        Vtr::FaceInterface fHood(fVerts.size());

        scheme.ComputeFaceVertexMask(fHood, fMask);

        //  Apply the weights to the parent face's vertices:
        dst[cVert].Clear();

        for (int i = 0; i < fVerts.size(); ++i) {

            dst[cVert].AddWithWeight(src[fVerts[i]], fVertWeights[i]);

            dst[cVert].AddVaryingWithWeight(src[fVerts[i]], fVaryingWeight);
        }
    }
}

template <Sdc::SchemeType SCHEME, class T, class U>
inline void
PrimvarRefiner::interpolateChildVertsFromEdges(
    Vtr::Refinement const & refinement, T const & src, U & dst) const {

    Sdc::Scheme<SCHEME> scheme(_refiner._subdivOptions);

    const Vtr::Level& parent = refinement.parent();
    const Vtr::Level& child  = refinement.child();

    Vtr::EdgeInterface eHood(parent);

    float                               eVertWeights[2];
    Vtr::internal::StackBuffer<float,8> eFaceWeights(parent.getMaxEdgeFaces());

    for (int edge = 0; edge < parent.getNumEdges(); ++edge) {

        Vtr::Index cVert = refinement.getEdgeChildVertex(edge);
        if (!Vtr::IndexIsValid(cVert))
            continue;

        //  Declare and compute mask weights for this vertex relative to its parent edge:
        ConstIndexArray eVerts = parent.getEdgeVertices(edge),
                        eFaces = parent.getEdgeFaces(edge);

        Vtr::MaskInterface eMask(eVertWeights, 0, eFaceWeights);

        eHood.SetIndex(edge);

        Sdc::Crease::Rule pRule = (parent.getEdgeSharpness(edge) > 0.0f) ? Sdc::Crease::RULE_CREASE : Sdc::Crease::RULE_SMOOTH;
        Sdc::Crease::Rule cRule = child.getVertexRule(cVert);

        scheme.ComputeEdgeVertexMask(eHood, eMask, pRule, cRule);

        //  Apply the weights to the parent edges's vertices and (if applicable) to
        //  the child vertices of its incident faces:
        dst[cVert].Clear();
        dst[cVert].AddWithWeight(src[eVerts[0]], eVertWeights[0]);
        dst[cVert].AddWithWeight(src[eVerts[1]], eVertWeights[1]);

        dst[cVert].AddVaryingWithWeight(src[eVerts[0]], 0.5f);
        dst[cVert].AddVaryingWithWeight(src[eVerts[1]], 0.5f);

        if (eMask.GetNumFaceWeights() > 0) {

            for (int i = 0; i < eFaces.size(); ++i) {

                if (eMask.AreFaceWeightsForFaceCenters()) {
                    assert(refinement.getNumChildVerticesFromFaces() > 0);
                    Vtr::Index cVertOfFace = refinement.getFaceChildVertex(eFaces[i]);

                    assert(Vtr::IndexIsValid(cVertOfFace));
                    dst[cVert].AddWithWeight(dst[cVertOfFace], eFaceWeights[i]);
                } else {
                    Vtr::Index            pFace      = eFaces[i];
                    ConstIndexArray pFaceEdges = parent.getFaceEdges(pFace),
                                    pFaceVerts = parent.getFaceVertices(pFace);

                    int eInFace = 0;
                    for ( ; pFaceEdges[eInFace] != edge; ++eInFace ) ;

                    int vInFace = eInFace + 2;
                    if (vInFace >= pFaceVerts.size()) vInFace -= pFaceVerts.size();

                    Vtr::Index pVertNext = pFaceVerts[vInFace];
                    dst[cVert].AddWithWeight(src[pVertNext], eFaceWeights[i]);
                }
            }
        }
    }
}

template <Sdc::SchemeType SCHEME, class T, class U>
inline void
PrimvarRefiner::interpolateChildVertsFromVerts(
    Vtr::Refinement const & refinement, T const & src, U & dst) const {

    Sdc::Scheme<SCHEME> scheme(_refiner._subdivOptions);

    const Vtr::Level& parent = refinement.parent();
    const Vtr::Level& child  = refinement.child();

    Vtr::VertexInterface vHood(parent, child);

    Vtr::internal::StackBuffer<float,32> weightBuffer(2*parent.getMaxValence());

    for (int vert = 0; vert < parent.getNumVertices(); ++vert) {

        Vtr::Index cVert = refinement.getVertexChildVertex(vert);
        if (!Vtr::IndexIsValid(cVert))
            continue;

        //  Declare and compute mask weights for this vertex relative to its parent edge:
        ConstIndexArray vEdges = parent.getVertexEdges(vert),
                        vFaces = parent.getVertexFaces(vert);

        float   vVertWeight,
              * vEdgeWeights = weightBuffer,
              * vFaceWeights = vEdgeWeights + vEdges.size();

        Vtr::MaskInterface vMask(&vVertWeight, vEdgeWeights, vFaceWeights);

        vHood.SetIndex(vert, cVert);

        Sdc::Crease::Rule pRule = parent.getVertexRule(vert);
        Sdc::Crease::Rule cRule = child.getVertexRule(cVert);

        scheme.ComputeVertexVertexMask(vHood, vMask, pRule, cRule);

        //  Apply the weights to the parent vertex, the vertices opposite its incident
        //  edges, and the child vertices of its incident faces:
        //
        //  In order to improve numerical precision, its better to apply smaller weights
        //  first, so begin with the face-weights followed by the edge-weights and the
        //  vertex weight last.
        dst[cVert].Clear();

        if (vMask.GetNumFaceWeights() > 0) {
            assert(vMask.AreFaceWeightsForFaceCenters());

            for (int i = 0; i < vFaces.size(); ++i) {

                Vtr::Index cVertOfFace = refinement.getFaceChildVertex(vFaces[i]);
                assert(Vtr::IndexIsValid(cVertOfFace));
                dst[cVert].AddWithWeight(dst[cVertOfFace], vFaceWeights[i]);
            }
        }
        if (vMask.GetNumEdgeWeights() > 0) {

            for (int i = 0; i < vEdges.size(); ++i) {

                ConstIndexArray eVerts = parent.getEdgeVertices(vEdges[i]);
                Vtr::Index pVertOppositeEdge = (eVerts[0] == vert) ? eVerts[1] : eVerts[0];

                dst[cVert].AddWithWeight(src[pVertOppositeEdge], vEdgeWeights[i]);
            }
        }
        dst[cVert].AddWithWeight(src[vert], vVertWeight);

        dst[cVert].AddVaryingWithWeight(src[vert], 1.0f);
    }
}

//
// Varying only interpolation
//

template <class T, class U>
inline void
PrimvarRefiner::InterpolateVarying(T const * src, U * dst) const {

    for (int level = 1; level <= _refiner.GetMaxLevel(); ++level) {

        InterpolateVarying(level, src, dst);

        src = dst;
        dst += _refiner.GetLevel(level).GetNumVertices();
    }
}

template <class T, class U>
inline void
PrimvarRefiner::InterpolateVarying(int level, T const & src, U & dst) const {

    assert(level>0 and level<=(int)_refiner._refinements.size());

    Vtr::Refinement const & refinement = _refiner.getRefinement(level-1);

    varyingInterpolateChildVertsFromFaces(refinement, src, dst);
    varyingInterpolateChildVertsFromEdges(refinement, src, dst);
    varyingInterpolateChildVertsFromVerts(refinement, src, dst);
}

template <class T, class U>
inline void
PrimvarRefiner::varyingInterpolateChildVertsFromFaces(
    Vtr::Refinement const & refinement, T const & src, U & dst) const {

    if (refinement.getNumChildVerticesFromFaces() == 0) return;

    const Vtr::Level& parent = refinement.parent();

    for (int face = 0; face < parent.getNumFaces(); ++face) {

        Vtr::Index cVert = refinement.getFaceChildVertex(face);
        if (!Vtr::IndexIsValid(cVert))
            continue;

        ConstIndexArray fVerts = parent.getFaceVertices(face);

        float fVaryingWeight = 1.0f / (float) fVerts.size();

        //  Apply the weights to the parent face's vertices:
        dst[cVert].Clear();

        for (int i = 0; i < fVerts.size(); ++i) {
            dst[cVert].AddVaryingWithWeight(src[fVerts[i]], fVaryingWeight);
        }
    }
}

template <class T, class U>
inline void
PrimvarRefiner::varyingInterpolateChildVertsFromEdges(
    Vtr::Refinement const & refinement, T const & src, U & dst) const {

    const Vtr::Level& parent = refinement.parent();

    for (int edge = 0; edge < parent.getNumEdges(); ++edge) {

        Vtr::Index cVert = refinement.getEdgeChildVertex(edge);
        if (!Vtr::IndexIsValid(cVert))
            continue;

        //  Declare and compute mask weights for this vertex relative to its parent edge:
        ConstIndexArray eVerts = parent.getEdgeVertices(edge);

        //  Apply the weights to the parent edges's vertices
        dst[cVert].Clear();

        dst[cVert].AddVaryingWithWeight(src[eVerts[0]], 0.5f);
        dst[cVert].AddVaryingWithWeight(src[eVerts[1]], 0.5f);
    }
}

template <class T, class U>
inline void
PrimvarRefiner::varyingInterpolateChildVertsFromVerts(
    Vtr::Refinement const & refinement, T const & src, U & dst) const {

    const Vtr::Level& parent = refinement.parent();

    for (int vert = 0; vert < parent.getNumVertices(); ++vert) {

        Vtr::Index cVert = refinement.getVertexChildVertex(vert);
        if (!Vtr::IndexIsValid(cVert))
            continue;

        //  Apply the weights to the parent vertex
        dst[cVert].Clear();
        dst[cVert].AddVaryingWithWeight(src[vert], 1.0f);
    }
}


//
// Face-varying only interpolation
//

template <class T, class U>
inline void
PrimvarRefiner::InterpolateFaceVarying(T const * src, U * dst, int channel) const {

    for (int level = 1; level <= _refiner.GetMaxLevel(); ++level) {

        InterpolateFaceVarying(level, src, dst, channel);

        src = dst;
        dst += _refiner.getLevel(level).getNumFVarValues();
    }
}

template <class T, class U>
inline void
PrimvarRefiner::InterpolateFaceVarying(int level, T const & src, U & dst, int channel) const {

    assert(level>0 and level<=(int)_refiner._refinements.size());

    Vtr::Refinement const & refinement = _refiner.getRefinement(level-1);

    switch (_refiner._subdivType) {
    case Sdc::SCHEME_CATMARK:
        faceVaryingInterpolateChildVertsFromFaces<Sdc::SCHEME_CATMARK>(refinement, src, dst, channel);
        faceVaryingInterpolateChildVertsFromEdges<Sdc::SCHEME_CATMARK>(refinement, src, dst, channel);
        faceVaryingInterpolateChildVertsFromVerts<Sdc::SCHEME_CATMARK>(refinement, src, dst, channel);
        break;
    case Sdc::SCHEME_LOOP:
        faceVaryingInterpolateChildVertsFromFaces<Sdc::SCHEME_LOOP>(refinement, src, dst, channel);
        faceVaryingInterpolateChildVertsFromEdges<Sdc::SCHEME_LOOP>(refinement, src, dst, channel);
        faceVaryingInterpolateChildVertsFromVerts<Sdc::SCHEME_LOOP>(refinement, src, dst, channel);
        break;
    case Sdc::SCHEME_BILINEAR:
        faceVaryingInterpolateChildVertsFromFaces<Sdc::SCHEME_BILINEAR>(refinement, src, dst, channel);
        faceVaryingInterpolateChildVertsFromEdges<Sdc::SCHEME_BILINEAR>(refinement, src, dst, channel);
        faceVaryingInterpolateChildVertsFromVerts<Sdc::SCHEME_BILINEAR>(refinement, src, dst, channel);
        break;
    }
}

template <Sdc::SchemeType SCHEME, class T, class U>
inline void
PrimvarRefiner::faceVaryingInterpolateChildVertsFromFaces(
    Vtr::Refinement const & refinement, T const & src, U & dst, int channel) const {

    if (refinement.getNumChildVerticesFromFaces() == 0) return;

    Sdc::Scheme<SCHEME> scheme(_refiner._subdivOptions);

    const Vtr::Level& parentLevel = refinement.parent();
    const Vtr::Level& childLevel  = refinement.child();

    const Vtr::FVarLevel& parentFVar = *parentLevel._fvarChannels[channel];
    const Vtr::FVarLevel& childFVar  = *childLevel._fvarChannels[channel];

    Vtr::internal::StackBuffer<float,16> fValueWeights(parentLevel.getMaxValence());

    for (int face = 0; face < parentLevel.getNumFaces(); ++face) {

        Vtr::Index cVert = refinement.getFaceChildVertex(face);
        if (!Vtr::IndexIsValid(cVert))
            continue;

        Vtr::Index cVertValue = childFVar.getVertexValueOffset(cVert);

        //  The only difference for face-varying here is that we get the values associated
        //  with each face-vertex directly from the FVarLevel, rather than using the parent
        //  face-vertices directly.  If any face-vertex has any sibling values, then we may
        //  get the wrong one using the face-vertex index directly.

        //  Declare and compute mask weights for this vertex relative to its parent face:
        ConstIndexArray fValues = parentFVar.getFaceValues(face);

        Vtr::MaskInterface fMask(fValueWeights, 0, 0);
        Vtr::FaceInterface fHood(fValues.size());

        scheme.ComputeFaceVertexMask(fHood, fMask);

        //  Apply the weights to the parent face's vertices:
        dst[cVertValue].Clear();

        for (int i = 0; i < fValues.size(); ++i) {
            dst[cVertValue].AddWithWeight(src[fValues[i]], fValueWeights[i]);
        }
    }
}

template <Sdc::SchemeType SCHEME, class T, class U>
inline void
PrimvarRefiner::faceVaryingInterpolateChildVertsFromEdges(
    Vtr::Refinement const & refinement, T const & src, U & dst, int channel) const {

    Sdc::Scheme<SCHEME> scheme(_refiner._subdivOptions);

    const Vtr::Level& parentLevel = refinement.parent();
    const Vtr::Level& childLevel  = refinement.child();

    const Vtr::FVarRefinement& refineFVar = *refinement._fvarChannels[channel];
    const Vtr::FVarLevel&      parentFVar = *parentLevel._fvarChannels[channel];
    const Vtr::FVarLevel&      childFVar  = *childLevel._fvarChannels[channel];

    //
    //  Allocate and intialize (if linearly interpolated) interpolation weights for
    //  the edge mask:
    //
    float                               eVertWeights[2];
    Vtr::internal::StackBuffer<float,8> eFaceWeights(parentLevel.getMaxEdgeFaces());

    Vtr::MaskInterface eMask(eVertWeights, 0, eFaceWeights);

    bool isLinearFVar = parentFVar._isLinear;
    if (isLinearFVar) {
        eMask.SetNumVertexWeights(2);
        eMask.SetNumEdgeWeights(0);
        eMask.SetNumFaceWeights(0);

        eVertWeights[0] = 0.5f;
        eVertWeights[1] = 0.5f;
    }

    Vtr::EdgeInterface eHood(parentLevel);

    for (int edge = 0; edge < parentLevel.getNumEdges(); ++edge) {

        Vtr::Index cVert = refinement.getEdgeChildVertex(edge);
        if (!Vtr::IndexIsValid(cVert))
            continue;

        ConstIndexArray cVertValues = childFVar.getVertexValues(cVert);

        bool fvarEdgeVertMatchesVertex = childFVar.valueTopologyMatches(cVertValues[0]);
        if (fvarEdgeVertMatchesVertex) {
            //
            //  If smoothly interpolated, compute new weights for the edge mask:
            //
            if (!isLinearFVar) {
                eHood.SetIndex(edge);

                Sdc::Crease::Rule pRule = (parentLevel.getEdgeSharpness(edge) > 0.0f)
                                        ? Sdc::Crease::RULE_CREASE : Sdc::Crease::RULE_SMOOTH;
                Sdc::Crease::Rule cRule = childLevel.getVertexRule(cVert);

                scheme.ComputeEdgeVertexMask(eHood, eMask, pRule, cRule);
            }

            //  Apply the weights to the parent edges's vertices and (if applicable) to
            //  the child vertices of its incident faces:
            //
            //  Even though the face-varying topology matches the vertex topology, we need
            //  to be careful here when getting values corresponding to the two end-vertices.
            //  While the edge may be continuous, the vertices at their ends may have
            //  discontinuities elsewhere in their neighborhood (i.e. on the "other side"
            //  of the end-vertex) and so have sibling values associated with them.  In most
            //  cases the topology for an end-vertex will match and we can use it directly,
            //  but we must still check and retrieve as needed.
            //
            //  Indices for values corresponding to face-vertices are guaranteed to match,
            //  so we can use the child-vertex indices directly.
            //
            //  And by "directly", we always use getVertexValue(vertexIndex) to reference
            //  values in the "src" to account for the possible indirection that may exist at
            //  level 0 -- where there may be fewer values than vertices and an additional
            //  indirection is necessary.  We can use a vertex index directly for "dst" when
            //  it matches.
            //
            Vtr::Index eVertValues[2];

            parentFVar.getEdgeFaceValues(edge, 0, eVertValues);

            Index cVertValue = cVertValues[0];

            dst[cVertValue].Clear();
            dst[cVertValue].AddWithWeight(src[eVertValues[0]], eVertWeights[0]);
            dst[cVertValue].AddWithWeight(src[eVertValues[1]], eVertWeights[1]);

            if (eMask.GetNumFaceWeights() > 0) {

                ConstIndexArray  eFaces = parentLevel.getEdgeFaces(edge);

                for (int i = 0; i < eFaces.size(); ++i) {
                    if (eMask.AreFaceWeightsForFaceCenters()) {

                        Vtr::Index cVertOfFace = refinement.getFaceChildVertex(eFaces[i]);
                        assert(Vtr::IndexIsValid(cVertOfFace));

                        Vtr::Index cValueOfFace = childFVar.getVertexValueOffset(cVertOfFace);
                        dst[cVertValue].AddWithWeight(dst[cValueOfFace], eFaceWeights[i]);
                    } else {
                        Vtr::Index            pFace      = eFaces[i];
                        ConstIndexArray pFaceEdges = parentLevel.getFaceEdges(pFace),
                                        pFaceVerts = parentLevel.getFaceVertices(pFace);

                        int eInFace = 0;
                        for ( ; pFaceEdges[eInFace] != edge; ++eInFace ) ;

                        //  Edge "i" spans vertices [i,i+1] so we want i+2...
                        int vInFace = eInFace + 2;
                        if (vInFace >= pFaceVerts.size()) vInFace -= pFaceVerts.size();

                        Vtr::Index pValueNext = parentFVar.getFaceValues(pFace)[vInFace];
                        dst[cVertValue].AddWithWeight(src[pValueNext], eFaceWeights[i]);
                    }
                }
            }
        } else {
            //
            //  Mismatched edge-verts should just be linearly interpolated between the pairs of
            //  values for each sibling of the child edge-vertex -- the question is:  which face
            //  holds that pair of values for a given sibling?
            //
            //  In the manifold case, the sibling and edge-face indices will correspond.  We
            //  will eventually need to update this to account for > 3 incident faces.
            //
            for (int i = 0; i < cVertValues.size(); ++i) {
                Vtr::Index eVertValues[2];
                int      eFaceIndex = refineFVar.getChildValueParentSource(cVert, i);
                assert(eFaceIndex == i);

                parentFVar.getEdgeFaceValues(edge, eFaceIndex, eVertValues);

                Index cVertValue = cVertValues[i];

                dst[cVertValue].Clear();
                dst[cVertValue].AddWithWeight(src[eVertValues[0]], 0.5);
                dst[cVertValue].AddWithWeight(src[eVertValues[1]], 0.5);
            }
        }
    }
}

template <Sdc::SchemeType SCHEME, class T, class U>
inline void
PrimvarRefiner::faceVaryingInterpolateChildVertsFromVerts(
    Vtr::Refinement const & refinement, T const & src, U & dst, int channel) const {

    Sdc::Scheme<SCHEME> scheme(_refiner._subdivOptions);

    const Vtr::Level& parentLevel = refinement.parent();
    const Vtr::Level& childLevel  = refinement.child();

    const Vtr::FVarRefinement& refineFVar = *refinement._fvarChannels[channel];
    const Vtr::FVarLevel&      parentFVar = *parentLevel._fvarChannels[channel];
    const Vtr::FVarLevel&      childFVar  = *childLevel._fvarChannels[channel];

    bool isLinearFVar = parentFVar._isLinear;

    Vtr::internal::StackBuffer<float,32> weightBuffer(2*parentLevel.getMaxValence());

    Vtr::internal::StackBuffer<Vtr::Index,16> vEdgeValues(parentLevel.getMaxValence());

    Vtr::VertexInterface vHood(parentLevel, childLevel);

    for (int vert = 0; vert < parentLevel.getNumVertices(); ++vert) {

        Vtr::Index cVert = refinement.getVertexChildVertex(vert);
        if (!Vtr::IndexIsValid(cVert))
            continue;

        ConstIndexArray pVertValues = parentFVar.getVertexValues(vert),
                        cVertValues = childFVar.getVertexValues(cVert);

        bool fvarVertVertMatchesVertex = childFVar.valueTopologyMatches(cVertValues[0]);
        if (isLinearFVar && fvarVertVertMatchesVertex) {
            dst[cVertValues[0]].Clear();
            dst[cVertValues[0]].AddWithWeight(src[pVertValues[0]], 1.0f);
            continue;
        }

        if (fvarVertVertMatchesVertex) {
            //
            //  Declare and compute mask weights for this vertex relative to its parent edge:
            //
            //  (We really need to encapsulate this somewhere else for use here and in the
            //  general case)
            //
            ConstIndexArray vEdges = parentLevel.getVertexEdges(vert);

            float   vVertWeight;
            float * vEdgeWeights = weightBuffer;
            float * vFaceWeights = vEdgeWeights + vEdges.size();

            Vtr::MaskInterface vMask(&vVertWeight, vEdgeWeights, vFaceWeights);

            vHood.SetIndex(vert, cVert);

            Sdc::Crease::Rule pRule = parentLevel.getVertexRule(vert);
            Sdc::Crease::Rule cRule = childLevel.getVertexRule(cVert);

            scheme.ComputeVertexVertexMask(vHood, vMask, pRule, cRule);

            //  Apply the weights to the parent vertex, the vertices opposite its incident
            //  edges, and the child vertices of its incident faces:
            //
            //  Even though the face-varying topology matches the vertex topology, we need
            //  to be careful here when getting values corresponding to vertices at the
            //  ends of edges.  While the edge may be continuous, the end vertex may have
            //  discontinuities elsewhere in their neighborhood (i.e. on the "other side"
            //  of the end-vertex) and so have sibling values associated with them.  In most
            //  cases the topology for an end-vertex will match and we can use it directly,
            //  but we must still check and retrieve as needed.
            //
            //  Indices for values corresponding to face-vertices are guaranteed to match,
            //  so we can use the child-vertex indices directly.
            //
            //  And by "directly", we always use getVertexValue(vertexIndex) to reference
            //  values in the "src" to account for the possible indirection that may exist at
            //  level 0 -- where there may be fewer values than vertices and an additional
            //  indirection is necessary.  We can use a vertex index directly for "dst" when
            //  it matches.
            //
            //  As with applying the mask to vertex data, in order to improve numerical
            //  precision, its better to apply smaller weights first, so begin with the
            //  face-weights followed by the edge-weights and the vertex weight last.
            //
            Vtr::Index pVertValue = pVertValues[0];
            Vtr::Index cVertValue = cVertValues[0];

            dst[cVertValue].Clear();
            if (vMask.GetNumFaceWeights() > 0) {
                assert(vMask.AreFaceWeightsForFaceCenters());

                ConstIndexArray vFaces = parentLevel.getVertexFaces(vert);

                for (int i = 0; i < vFaces.size(); ++i) {

                    Vtr::Index cVertOfFace  = refinement.getFaceChildVertex(vFaces[i]);
                    assert(Vtr::IndexIsValid(cVertOfFace));

                    Vtr::Index cValueOfFace = childFVar.getVertexValueOffset(cVertOfFace);
                    dst[cVertValue].AddWithWeight(dst[cValueOfFace], vFaceWeights[i]);
                }
            }
            if (vMask.GetNumEdgeWeights() > 0) {

                parentFVar.getVertexEdgeValues(vert, vEdgeValues);

                for (int i = 0; i < vEdges.size(); ++i) {
                    dst[cVertValue].AddWithWeight(src[vEdgeValues[i]], vEdgeWeights[i]);
                }
            }
            dst[cVertValue].AddWithWeight(src[pVertValue], vVertWeight);
        } else {
            //
            //  Each FVar value associated with a vertex will be either a corner or a crease,
            //  or potentially in transition from corner to crease:
            //      - if the CHILD is a corner, there can be no transition so we have a corner
            //      - otherwise if the PARENT is a crease, both will be creases (no transition)
            //      - otherwise the parent must be a corner and the child a crease (transition)
            //
            Vtr::FVarLevel::ConstValueTagArray pValueTags = parentFVar.getVertexValueTags(vert);
            Vtr::FVarLevel::ConstValueTagArray cValueTags = childFVar.getVertexValueTags(cVert);

            for (int cSibling = 0; cSibling < cVertValues.size(); ++cSibling) {
                int pSibling = refineFVar.getChildValueParentSource(cVert, cSibling);
                assert(pSibling == cSibling);

                Vtr::Index pVertValue = pVertValues[pSibling];
                Vtr::Index cVertValue = cVertValues[cSibling];

                dst[cVertValue].Clear();
                if (cValueTags[cSibling].isCorner()) {
                    dst[cVertValue].AddWithWeight(src[pVertValue], 1.0f);
                } else {
                    //
                    //  We have either a crease or a transition from corner to crease -- in
                    //  either case, we need the end values for the full/fractional crease:
                    //
                    Index pEndValues[2];
                    parentFVar.getVertexCreaseEndValues(vert, pSibling, pEndValues);

                    float vWeight = 0.75f;
                    float eWeight = 0.125f;

                    //
                    //  If semisharp we need to apply fractional weighting -- if made sharp because
                    //  of the other sibling (dependent-sharp) use the fractional weight from that
                    //  other sibling (should only occur when there are 2):
                    //
                    if (pValueTags[pSibling].isSemiSharp()) {
                        float wCorner = pValueTags[pSibling].isDepSharp()
                                      ? refineFVar.getFractionalWeight(vert, !pSibling, cVert, !cSibling)
                                      : refineFVar.getFractionalWeight(vert, pSibling, cVert, cSibling);
                        float wCrease = 1.0f - wCorner;

                        vWeight = wCrease * 0.75f + wCorner;
                        eWeight = wCrease * 0.125f;
                    }
                    dst[cVertValue].AddWithWeight(src[pEndValues[0]], eWeight);
                    dst[cVertValue].AddWithWeight(src[pEndValues[1]], eWeight);
                    dst[cVertValue].AddWithWeight(src[pVertValue], vWeight);
                }
            }
        }
    }
}

template <class T, class U>
inline void
PrimvarRefiner::Limit(T const & src, U & dst) const {

    if (_refiner.getLevel(_refiner.GetMaxLevel()).getNumVertexEdgesTotal() == 0) {
        Error(FAR_RUNTIME_ERROR,
            "Cannot compute limit points -- last level of refinement does not include full topology.");
        return;
    }

    switch (_refiner._subdivType) {
    case Sdc::SCHEME_CATMARK:
        limit<Sdc::SCHEME_CATMARK>(src, dst, (U*)0, (U*)0);
        break;
    case Sdc::SCHEME_LOOP:
        limit<Sdc::SCHEME_LOOP>(src, dst, (U*)0, (U*)0);
        break;
    case Sdc::SCHEME_BILINEAR:
        limit<Sdc::SCHEME_BILINEAR>(src, dst, (U*)0, (U*)0);
        break;
    }
}

template <class T, class U, class U1, class U2>
inline void
PrimvarRefiner::Limit(T const & src, U & dstPos, U1 & dstTan1, U2 & dstTan2) const {

    if (_refiner.getLevel(_refiner.GetMaxLevel()).getNumVertexEdgesTotal() == 0) {
        Error(FAR_RUNTIME_ERROR,
            "Cannot compute limit points -- last level of refinement does not include full topology.");
        return;
    }

    switch (_refiner._subdivType) {
    case Sdc::SCHEME_CATMARK:
        limit<Sdc::SCHEME_CATMARK>(src, dstPos, &dstTan1, &dstTan2);
        break;
    case Sdc::SCHEME_LOOP:
        limit<Sdc::SCHEME_LOOP>(src, dstPos, &dstTan1, &dstTan2);
        break;
    case Sdc::SCHEME_BILINEAR:
        limit<Sdc::SCHEME_BILINEAR>(src, dstPos, &dstTan1, &dstTan2);
        break;
    }
}

template <Sdc::SchemeType SCHEME, class T, class U, class U1, class U2>
inline void
PrimvarRefiner::limit(T const & src, U & dstPos, U1 * dstTan1Ptr, U2 * dstTan2Ptr) const {

    Sdc::Scheme<SCHEME> scheme(_refiner._subdivOptions);

    Vtr::Level const & level = _refiner.getLevel(_refiner.GetMaxLevel());

    int  maxWeightsPerMask = 1 + 2 * level.getMaxValence();
    bool hasTangents = (dstTan1Ptr && dstTan2Ptr);
    int  numMasks = 1 + (hasTangents ? 2 : 0);

    Vtr::internal::StackBuffer<Index,33> indexBuffer(maxWeightsPerMask);
    Vtr::internal::StackBuffer<float,99> weightBuffer(numMasks * maxWeightsPerMask);

    float * vPosWeights = weightBuffer,
          * ePosWeights = vPosWeights + 1,
          * fPosWeights = ePosWeights + level.getMaxValence();
    float * vTan1Weights = vPosWeights + maxWeightsPerMask,
          * eTan1Weights = ePosWeights + maxWeightsPerMask,
          * fTan1Weights = fPosWeights + maxWeightsPerMask;
    float * vTan2Weights = vTan1Weights + maxWeightsPerMask,
          * eTan2Weights = eTan1Weights + maxWeightsPerMask,
          * fTan2Weights = fTan1Weights + maxWeightsPerMask;

    Vtr::MaskInterface posMask( vPosWeights,  ePosWeights,  fPosWeights);
    Vtr::MaskInterface tan1Mask(vTan1Weights, eTan1Weights, fTan1Weights);
    Vtr::MaskInterface tan2Mask(vTan2Weights, eTan2Weights, fTan2Weights);

    //  This is a bit obscure -- assigning both parent and child as last level -- but
    //  this mask type was intended for another purpose.  Consider one for the limit:
    Vtr::VertexInterface vHood(level, level);

    for (int vert = 0; vert < level.getNumVertices(); ++vert) {
        ConstIndexArray vEdges = level.getVertexEdges(vert);

        //  Incomplete vertices (present in sparse refinement) do not have their full
        //  topological neighborhood to determine a proper limit -- just leave the
        //  vertex at the refined location and continue to the next:
        if (level._vertTags[vert]._incomplete || (vEdges.size() == 0)) {
            dstPos[vert].Clear();
            dstPos[vert].AddWithWeight(src[vert], 1.0);
            if (hasTangents) {
                (*dstTan1Ptr)[vert].Clear();
                (*dstTan2Ptr)[vert].Clear();
            }
            continue;
        }

        //
        //  Limit masks require the subdivision Rule for the vertex in order to deal
        //  with infinitely sharp features correctly -- including boundaries and corners.
        //  The vertex neighborhood is minimally defined with vertex and edge counts.
        //
        Sdc::Crease::Rule vRule = level.getVertexRule(vert);

        //  This is a bit obscure -- child vertex index will be ignored here
        vHood.SetIndex(vert, vert);

        if (hasTangents) {
            scheme.ComputeVertexLimitMask(vHood, posMask, tan1Mask, tan2Mask, vRule);
        } else {
            scheme.ComputeVertexLimitMask(vHood, posMask, vRule);
        }

        //
        //  Gather the neighboring vertices of this vertex -- the vertices opposite its
        //  incident edges, and the opposite vertices of its incident faces:
        //
        Index * eIndices = indexBuffer;
        Index * fIndices = indexBuffer + vEdges.size();

        for (int i = 0; i < vEdges.size(); ++i) {
            ConstIndexArray eVerts = level.getEdgeVertices(vEdges[i]);

            eIndices[i] = (eVerts[0] == vert) ? eVerts[1] : eVerts[0];
        }
        if (posMask.GetNumFaceWeights() || (hasTangents && tan1Mask.GetNumFaceWeights())) {
            ConstIndexArray      vFaces = level.getVertexFaces(vert);
            ConstLocalIndexArray vInFace = level.getVertexFaceLocalIndices(vert);

            for (int i = 0; i < vFaces.size(); ++i) {
                ConstIndexArray fVerts = level.getFaceVertices(vFaces[i]);

                LocalIndex vOppInFace = (vInFace[i] + 2);
                if (vOppInFace >= fVerts.size()) vOppInFace -= (LocalIndex)fVerts.size();

                fIndices[i] = level.getFaceVertices(vFaces[i])[vOppInFace];
            }
        }

        //
        //  Combine the weights and indices for position and tangents.  As with applying
        //  refinment masks to vertex data, in order to improve numerical precision, its
        //  better to apply smaller weights first, so begin with the face-weights followed
        //  by the edge-weights and the vertex weight last.
        //
        dstPos[vert].Clear();
        for (int i = 0; i < posMask.GetNumFaceWeights(); ++i) {
            dstPos[vert].AddWithWeight(src[fIndices[i]], fPosWeights[i]);
        }
        for (int i = 0; i < posMask.GetNumEdgeWeights(); ++i) {
            dstPos[vert].AddWithWeight(src[eIndices[i]], ePosWeights[i]);
        }
        dstPos[vert].AddWithWeight(src[vert], vPosWeights[0]);

        //
        //  Apply the tangent masks -- both will have the same number of weights and 
        //  indices (one tangent may be "padded" to accomodate the other), but these
        //  may differ from those of the position:
        //
        if (hasTangents) {
            assert(tan1Mask.GetNumFaceWeights() == tan2Mask.GetNumFaceWeights());
            assert(tan1Mask.GetNumEdgeWeights() == tan2Mask.GetNumEdgeWeights());

            U1 & dstTan1 = *dstTan1Ptr;
            U2 & dstTan2 = *dstTan2Ptr;

            dstTan1[vert].Clear();
            dstTan2[vert].Clear();
            for (int i = 0; i < tan1Mask.GetNumFaceWeights(); ++i) {
                dstTan1[vert].AddWithWeight(src[fIndices[i]], fTan1Weights[i]);
                dstTan2[vert].AddWithWeight(src[fIndices[i]], fTan2Weights[i]);
            }
            for (int i = 0; i < tan1Mask.GetNumEdgeWeights(); ++i) {
                dstTan1[vert].AddWithWeight(src[eIndices[i]], eTan1Weights[i]);
                dstTan2[vert].AddWithWeight(src[eIndices[i]], eTan2Weights[i]);
            }
            dstTan1[vert].AddWithWeight(src[vert], vTan1Weights[0]);
            dstTan2[vert].AddWithWeight(src[vert], vTan2Weights[0]);
        }
    }
}

template <class T, class U>
inline void
PrimvarRefiner::LimitFaceVarying(T const & src, U * dst, int channel) const {

    if (_refiner.getLevel(_refiner.GetMaxLevel()).getNumVertexEdgesTotal() == 0) {
        Error(FAR_RUNTIME_ERROR,
            "Cannot compute limit points -- last level of refinement does not include full topology.");
        return;
    }

    switch (_refiner._subdivType) {
    case Sdc::SCHEME_CATMARK:
        faceVaryingLimit<Sdc::SCHEME_CATMARK>(src, dst, channel);
        break;
    case Sdc::SCHEME_LOOP:
        faceVaryingLimit<Sdc::SCHEME_LOOP>(src, dst, channel);
        break;
    case Sdc::SCHEME_BILINEAR:
        faceVaryingLimit<Sdc::SCHEME_BILINEAR>(src, dst, channel);
        break;
    }
}

template <Sdc::SchemeType SCHEME, class T, class U>
inline void
PrimvarRefiner::faceVaryingLimit(T const & src, U * dst, int channel) const {

    Sdc::Scheme<SCHEME> scheme(_refiner._subdivOptions);

    Vtr::Level const &      level       = _refiner.getLevel(_refiner.GetMaxLevel());
    Vtr::FVarLevel const &  fvarChannel = *level._fvarChannels[channel];

    int maxWeightsPerMask = 1 + 2 * level.getMaxValence();

    Vtr::internal::StackBuffer<float,33> weightBuffer(maxWeightsPerMask);
    Vtr::internal::StackBuffer<Index,16> vEdgeBuffer(level.getMaxValence());

    //  This is a bit obscure -- assign both parent and child as last level
    Vtr::VertexInterface vHood(level, level);

    for (int vert = 0; vert < level.getNumVertices(); ++vert) {

        ConstIndexArray vEdges  = level.getVertexEdges(vert);
        ConstIndexArray vValues = fvarChannel.getVertexValues(vert);

        //  Incomplete vertices (present in sparse refinement) do not have their full
        //  topological neighborhood to determine a proper limit -- just leave the
        //  values (perhaps more than one per vertex) at the refined location.
        //
        //  The same can be done if the face-varying channel is purely linear.
        //
        bool isIncomplete = (level._vertTags[vert]._incomplete || (vEdges.size() == 0));
        if (isIncomplete || fvarChannel._isLinear) {
            for (int i = 0; i < vValues.size(); ++i) {
                Vtr::Index vValue = vValues[i];

                dst[vValue].Clear();
                dst[vValue].AddWithWeight(src[vValue], 1.0f);
            }
            continue;
        }

        bool fvarVertMatchesVertex = fvarChannel.valueTopologyMatches(vValues[0]);
        if (fvarVertMatchesVertex) {

            //  Assign the mask weights to the common buffer and compute the mask:
            //
            float * vWeights = weightBuffer,
                  * eWeights = vWeights + 1,
                  * fWeights = eWeights + vEdges.size();

            Vtr::MaskInterface vMask(vWeights, eWeights, fWeights);

            vHood.SetIndex(vert, vert);

            scheme.ComputeVertexLimitMask(vHood, vMask, level.getVertexRule(vert));

            //
            //  Apply mask to corresponding FVar values for neighboring vertices:
            //
            Vtr::Index vValue = vValues[0];

            dst[vValue].Clear();
            if (vMask.GetNumFaceWeights() > 0) {
                assert(!vMask.AreFaceWeightsForFaceCenters());

                ConstIndexArray      vFaces = level.getVertexFaces(vert);
                ConstLocalIndexArray vInFace = level.getVertexFaceLocalIndices(vert);

                for (int i = 0; i < vFaces.size(); ++i) {
                    ConstIndexArray faceValues = fvarChannel.getFaceValues(vFaces[i]);
                    LocalIndex vOppInFace = vInFace[i] + 2;
                    if (vOppInFace >= faceValues.size()) vOppInFace -= faceValues.size();

                    Index vValueOppositeFace = faceValues[vOppInFace];

                    dst[vValue].AddWithWeight(src[vValueOppositeFace], fWeights[i]);
                }
            }
            if (vMask.GetNumEdgeWeights() > 0) {
                Index * vEdgeValues = vEdgeBuffer;
                fvarChannel.getVertexEdgeValues(vert, vEdgeValues);

                for (int i = 0; i < vEdges.size(); ++i) {
                    dst[vValue].AddWithWeight(src[vEdgeValues[i]], eWeights[i]);
                }
            }
            dst[vValue].AddWithWeight(src[vValue], vWeights[0]);
        } else {
            //
            //  Sibling FVar values associated with a vertex will be either a corner or a crease:
            //
            for (int i = 0; i < vValues.size(); ++i) {
                Vtr::Index vValue = vValues[i];

                dst[vValue].Clear();
                if (fvarChannel.getValueTag(vValue).isCorner()) {
                    dst[vValue].AddWithWeight(src[vValue], 1.0f);
                } else {
                    Index vEndValues[2];
                    fvarChannel.getVertexCreaseEndValues(vert, i, vEndValues);

                    dst[vValue].AddWithWeight(src[vEndValues[0]], 1.0f/6.0f);
                    dst[vValue].AddWithWeight(src[vEndValues[1]], 1.0f/6.0f);
                    dst[vValue].AddWithWeight(src[vValue], 2.0f/3.0f);
                }
            }
        }
    }
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;
} // end namespace OpenSubdiv

#endif /* OPENSUBDIV3_FAR_PRIMVAR_REFINER_H */
