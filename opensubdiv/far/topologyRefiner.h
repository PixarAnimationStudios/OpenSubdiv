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
#ifndef FAR_TOPOLOGY_REFINER_H
#define FAR_TOPOLOGY_REFINER_H

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
#include "../vtr/maskInterfaces.h"
#include "../far/types.h"

#include <vector>
#include <cassert>
#include <cstdio>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Vtr { class SparseSelector; }

namespace Far {

template <class MESH> class TopologyRefinerFactory;

///
///  \brief Stores topology data for a specified set of refinement options.
///
class TopologyRefiner {

public:

    /// \brief Constructor
    TopologyRefiner(Sdc::SchemeType type, Sdc::Options options = Sdc::Options());

    /// \brief Destructor
    ~TopologyRefiner();

    /// \brief Returns the subdivision scheme
    Sdc::SchemeType GetSchemeType() const    { return _subdivType; }

    /// \brief Returns the subdivision options
    Sdc::Options GetSchemeOptions() const { return _subdivOptions; }

    /// \brief Returns true if uniform subdivision has been applied
    bool IsUniform() const   { return _isUniform; }

    /// \brief Returns the number of refinement levels
    int  GetNumLevels() const { return (int)_levels.size(); }

    /// \brief Returns the highest level of refinement
    int  GetMaxLevel() const { return _maxLevel; }

    /// \ brief Returns true if faces have been tagged as holes
    bool HasHoles() const { return _hasHoles; }

    // XXXX barfowl -- should cache these internally for trivial return)

    /// \brief Returns the total number of vertices in all levels
    int GetNumVerticesTotal() const;

    /// \brief Returns the total number of edges in all levels
    int GetNumEdgesTotal() const;

    /// \brief Returns the total number of edges in all levels
    int GetNumFacesTotal() const;

    /// \brief Returns the total number of face vertices in all levels
    int GetNumFaceVerticesTotal() const;

    //@{
    ///  @name High-level refinement and related methods
    ///

    //
    // Uniform refinement
    //

    /// \brief Uniform refinement options
    struct UniformOptions {

        UniformOptions(int level) :
            refinementLevel(level),
            fullTopologyInLastLevel(false) { }

        unsigned int refinementLevel:4,         ///< Number of refinement iterations
                     fullTopologyInLastLevel:1; ///< Skip secondary topological relationships
                                                ///< at the highest level of refinement.
    };

    /// \brief Refine the topology uniformly
    ///
    /// @param options   Options controlling uniform refinement
    ///
    void RefineUniform(UniformOptions options);

    //
    // Adaptive refinement
    //

    /// \brief Adaptive refinement options
    struct AdaptiveOptions {

        AdaptiveOptions(int level) :
            isolationLevel(level),
            fullTopologyInLastLevel(false),
            useSingleCreasePatch(false) { }

        unsigned int isolationLevel:4,          ///< Number of iterations applied to isolate
                                                ///< extraordinary vertices and creases
                     fullTopologyInLastLevel:1, ///< Skip secondary topological relationships
                                                ///< at the highest level of refinement.
                     useSingleCreasePatch:1;    ///< Use 'single-crease' patch and stop
                                                ///< isolation where applicable
    };

    /// \brief Feature Adaptive topology refinement
    ///
    /// @param options   Options controlling adaptive refinement
    ///
    void RefineAdaptive(AdaptiveOptions options);

    /// \brief Unrefine the topology (keep control cage)
    void Unrefine();

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
    ///       See the <a href=http://internal-graphics.pixar.com/opensubdiv/docs/tutorials.html>
    ///       Far tutorials</a> for code examples.
    ///

    /// \brief Apply vertex and varying interpolation weights to a primvar
    ///        buffer
    ///
    /// The destination buffer must allocate an array of data for all the
    /// refined vertices (at least GetNumVerticesTotal()-GetNumVertices(0))
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
    /// refined vertices (at least GetNumVertices(level))
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
    /// The destination buffer must allocate an array of data for all the
    /// refined vertices (at least GetNumVerticesTotal()-GetNumVertices(0))
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
    /// refined vertices (at least GetNumVertices(level))
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
    /// (at least GetNumVertices(GetMaxLevel()))
    ///
    /// @param src  Source primvar buffer (refined vertex data) for last level
    ///
    /// @param dst  Destination primvar buffer (vertex data at the limit)
    ///
    template <class T, class U> void Limit(T const & src, U * dst) const;

    //@}

    //@{
    /// @name Inspection of components per level
    ///


    /// \brief Returns the number of vertices at a given level of refinement
    int GetNumVertices(int level) const {
        return _levels[level]->getNumVertices();
    }

    /// \brief Returns the number of edges at a given level of refinement
    int GetNumEdges(int level) const {
        return _levels[level]->getNumEdges();
    }

    /// \brief Returns the number of face vertex indices at a given level of refinement
    int GetNumFaces(int level) const {
        return _levels[level]->getNumFaces();
    }

    /// \brief Returns the number of faces marked as holes at the given level
    int GetNumHoles(int level) const;

    /// \brief Returns the number of faces at a given level of refinement
    int GetNumFaceVertices(int level) const {
        return _levels[level]->getNumFaceVerticesTotal();
    }

    /// \brief Returns the sharpness of a given edge (at 'level' of refinement)
    float GetEdgeSharpness(int level, Index edge) const {
        return _levels[level]->getEdgeSharpness(edge);
    }

    /// \brief Returns the sharpness of a given vertex (at 'level' of refinement)
    float GetVertexSharpness(int level, Index vert) const {
        return _levels[level]->getVertexSharpness(vert);
    }

    /// \brief Returns the subdivision rule of a given vertex (at 'level' of refinement)
    Sdc::Crease::Rule GetVertexRule(int level, Index vert) const {
        return _levels[level]->getVertexRule(vert);
    }

    //@}

    //@{
    /// @name Topological relations -- incident/adjacent components
    ///


    /// \brief Returns the vertices of a 'face' at 'level'
    ConstIndexArray GetFaceVertices(int level, Index face) const {
        return _levels[level]->getFaceVertices(face);
    }

    /// \brief Returns the edges of a 'face' at 'level'
    ConstIndexArray GetFaceEdges(int level, Index face) const {
        return _levels[level]->getFaceEdges(face);
    }

    /// \brief Returns true if 'face' at 'level' is tagged as a hole
    bool IsHole(int level, Index face) const {
        return _levels[level]->isHole(face);
    }

    /// \brief Returns the vertices of an 'edge' at 'level' (2 of them)
    ConstIndexArray GetEdgeVertices(int level, Index edge) const {
        return _levels[level]->getEdgeVertices(edge);
    }

    /// \brief Returns the faces incident to 'edge' at 'level'
    ConstIndexArray GetEdgeFaces(int level, Index edge) const {
        return _levels[level]->getEdgeFaces(edge);
    }

    /// \brief Returns the faces incident to 'vertex' at 'level'
    ConstIndexArray GetVertexFaces(int level, Index vert) const {
        return _levels[level]->getVertexFaces(vert);
    }

    /// \brief Returns the edges incident to 'vertex' at 'level'
    ConstIndexArray GetVertexEdges(int level, Index vert) const {
        return _levels[level]->getVertexEdges(vert);
    }

    /// \brief Returns the local face indices of vertex 'vert' at 'level'
    ConstLocalIndexArray VertexFaceLocalIndices(int level, Index vert) const {
        return _levels[level]->getVertexFaceLocalIndices(vert);
    }

    /// \brief Returns the local edge indices of vertex 'vert' at 'level'
    ConstLocalIndexArray VertexEdgeLocalIndices(int level, Index vert) const {
        return _levels[level]->getVertexEdgeLocalIndices(vert);
    }

    bool FaceIsRegular(int level, Index face) const {
        ConstIndexArray fVerts = _levels[level]->getFaceVertices(face);
        Vtr::Level::VTag compFaceVertTag =
            _levels[level]->getFaceCompositeVTag(fVerts);
        return not compFaceVertTag._xordinary;
    }

    /// \brief Returns the edge with vertices 'v0' and 'v1' (or INDEX_INVALID if
    ///  they are not connected)
    Index FindEdge(int level, Index v0, Index v1) const {
        return _levels[level]->findEdge(v0, v1);
    }

    //@}

    //@{
    /// @name Inspection of face-varying channels and their contents:
    ///


    /// \brief Returns the number of face-varying channels in the tables
    int GetNumFVarChannels() const {
        return _levels[0]->getNumFVarChannels();
    }

    /// \brief Returns the total number of face-varying values in all levels
    int GetNumFVarValuesTotal(int channel = 0) const;

    /// \brief Returns the number of face-varying values at a given level of refinement
    int GetNumFVarValues(int level, int channel = 0) const {
        return _levels[level]->getNumFVarValues(channel);
    }

    /// \brief Returns the face-varying values of a 'face' at 'level'
    ConstIndexArray const GetFVarFaceValues(int level, Index face, int channel = 0) const {
        return _levels[level]->getFVarFaceValues(face, channel);
    }

    //@}

    //@{
    /// @name Parent-to-child relationships,
    /// Telationships between components in one level
    /// and the next (entries may be invalid if sparse):
    ///


    /// \brief Returns the child faces of face 'f' at 'level'
    ConstIndexArray GetFaceChildFaces(int level, Index f) const {
        return _refinements[level]->getFaceChildFaces(f);
    }

    /// \brief Returns the child edges of face 'f' at 'level'
    ConstIndexArray GetFaceChildEdges(int level, Index f) const {
        return _refinements[level]->getFaceChildEdges(f);
    }

    /// \brief Returns the child edges of edge 'e' at 'level'
    ConstIndexArray GetEdgeChildEdges(int level, Index e) const {
        return _refinements[level]->getEdgeChildEdges(e);
    }

    /// \brief Returns the child vertex of face 'f' at 'level'
    Index GetFaceChildVertex(  int level, Index f) const {
        return _refinements[level]->getFaceChildVertex(f);
    }

    /// \brief Returns the child vertex of edge 'e' at 'level'
    Index GetEdgeChildVertex(  int level, Index e) const {
        return _refinements[level]->getEdgeChildVertex(e);
    }

    /// \brief Returns the child vertex of vertex 'v' at 'level'
    Index GetVertexChildVertex(int level, Index v) const {
        return _refinements[level]->getVertexChildVertex(v);
    }

    //@}


    //@{
    /// Ptex
    ///

    /// \brief Returns the number of ptex faces in the mesh
    int GetNumPtexFaces() const;

    /// \brief Returns the ptex face index given a coarse face 'f' or -1
    int GetPtexIndex(Index f) const;

    /// \brief Returns ptex face adjacency information for a given coarse face
    ///
    /// @param face      coarse face index
    ///
    /// @param quadrant  quadrant index if 'face' is not a quad (the local ptex
    //                   sub-face index). Must be less than the number of face
    //                   vertices.
    ///
    /// @param adjFaces  ptex face indices of adjacent faces
    ///
    /// @param adjEdges  ptex edge indices of adjacent faces
    ///
    void GetPtexAdjacency(int face, int quadrant,
        int adjFaces[4], int adjEdges[4]) const;

    //@}

    //@{
    /// @name Debugging aides
    ///

    /// \brief Returns true if the topology of 'level' is valid
    bool ValidateTopology(int level) const {
        return _levels[level]->validateTopology();
    }

    /// \brief Prints topology information to console
    void PrintTopology(int level, bool children = true) const {
        _levels[level]->print(children ? _refinements[level] : 0);
    }

    //@}

protected:

    //
    //  For use by the TopologyRefinerFactory<MESH> subclasses to construct the base level:
    //
    template <class MESH>
    friend class TopologyRefinerFactory;

    //  Topology sizing methods required before allocation:
    void setNumBaseFaces(   int count) { _levels[0]->resizeFaces(count); }
    void setNumBaseEdges(   int count) { _levels[0]->resizeEdges(count); }
    void setNumBaseVertices(int count) { _levels[0]->resizeVertices(count); }

    void setNumBaseFaceVertices(Index f, int count) { _levels[0]->resizeFaceVertices(f, count); }
    void setNumBaseEdgeFaces(   Index e, int count) { _levels[0]->resizeEdgeFaces(e, count); }
    void setNumBaseVertexFaces( Index v, int count) { _levels[0]->resizeVertexFaces(v, count); }
    void setNumBaseVertexEdges( Index v, int count) { _levels[0]->resizeVertexEdges(v, count); }

    //  Topology assignment methods to populate base level after allocation:
    IndexArray setBaseFaceVertices(Index f) { return _levels[0]->getFaceVertices(f); }
    IndexArray setBaseFaceEdges(   Index f) { return _levels[0]->getFaceEdges(f); }
    IndexArray setBaseEdgeVertices(Index e) { return _levels[0]->getEdgeVertices(e); }
    IndexArray setBaseEdgeFaces(   Index e) { return _levels[0]->getEdgeFaces(e); }
    IndexArray setBaseVertexFaces( Index v) { return _levels[0]->getVertexFaces(v); }
    IndexArray setBaseVertexEdges( Index v) { return _levels[0]->getVertexEdges(v); }

    LocalIndexArray setBaseVertexFaceLocalIndices(Index v) { return _levels[0]->getVertexFaceLocalIndices(v); }
    LocalIndexArray setBaseVertexEdgeLocalIndices(Index v) { return _levels[0]->getVertexEdgeLocalIndices(v); }

    void populateBaseLocalIndices() { _levels[0]->populateLocalIndices(); }

    void setBaseEdgeNonManifold(Index e, bool b) { _levels[0]->setEdgeNonManifold(e, b); }
    void setBaseVertexNonManifold(Index v, bool b) { _levels[0]->setVertexNonManifold(v, b); }

    //  Optional feature tagging methods for setting sharpness, holes, etc.:
    void setBaseEdgeSharpness(Index e, float s)   { _levels[0]->getEdgeSharpness(e) = s; }
    void setBaseVertexSharpness(Index v, float s) { _levels[0]->getVertexSharpness(v) = s; }

    void setBaseFaceHole(Index f, bool b) { _levels[0]->setHole(f, b); _hasHoles |= b; }

    //  Optional methods for creating and assigning face-varying data channels:
    int createBaseFVarChannel(int numValues)                              { return _levels[0]->createFVarChannel(numValues, _subdivOptions); }
    int createBaseFVarChannel(int numValues, Sdc::Options const& options) { return _levels[0]->createFVarChannel(numValues, options); }

    IndexArray setBaseFVarFaceValues(Index face, int channel = 0) { return _levels[0]->getFVarFaceValues(face, channel); }

protected:

    //
    //  Lower level protected methods intended stricty for internal use:
    //
    friend class TopologyRefinerFactoryBase;
    friend class PatchTablesFactory;
    friend class GregoryBasisFactory;

    Vtr::Level       & getLevel(int l)       { return *_levels[l]; }
    Vtr::Level const & getLevel(int l) const { return *_levels[l]; }

    Vtr::Refinement       & getRefinement(int l)       { return *_refinements[l]; }
    Vtr::Refinement const & getRefinement(int l) const { return *_refinements[l]; }

private:
    void selectFeatureAdaptiveComponents(Vtr::SparseSelector& selector);

    template <Sdc::SchemeType SCHEME, class T, class U> void interpolateChildVertsFromFaces(Vtr::Refinement const &, T const & src, U & dst) const;
    template <Sdc::SchemeType SCHEME, class T, class U> void interpolateChildVertsFromEdges(Vtr::Refinement const &, T const & src, U & dst) const;
    template <Sdc::SchemeType SCHEME, class T, class U> void interpolateChildVertsFromVerts(Vtr::Refinement const &, T const & src, U & dst) const;

    template <class T, class U> void varyingInterpolateChildVertsFromFaces(Vtr::Refinement const &, T const & src, U & dst) const;
    template <class T, class U> void varyingInterpolateChildVertsFromEdges(Vtr::Refinement const &, T const & src, U & dst) const;
    template <class T, class U> void varyingInterpolateChildVertsFromVerts(Vtr::Refinement const &, T const & src, U & dst) const;

    template <Sdc::SchemeType SCHEME, class T, class U> void faceVaryingInterpolateChildVertsFromFaces(Vtr::Refinement const &, T const & src, U & dst, int channel) const;
    template <Sdc::SchemeType SCHEME, class T, class U> void faceVaryingInterpolateChildVertsFromEdges(Vtr::Refinement const &, T const & src, U & dst, int channel) const;
    template <Sdc::SchemeType SCHEME, class T, class U> void faceVaryingInterpolateChildVertsFromVerts(Vtr::Refinement const &, T const & src, U & dst, int channel) const;

    template <Sdc::SchemeType SCHEME, class T, class U> void limit(T const & src, U * dst) const;

    template <Sdc::SchemeType SCHEME, class T, class U> void faceVaryingLimit(T const & src, U * dst, int channel) const;

    void initializePtexIndices() const;

private:

    Sdc::SchemeType _subdivType;
    Sdc::Options    _subdivOptions;

    unsigned int _isUniform : 1,
                 _hasHoles : 1,
                 _useSingleCreasePatch : 1,
                 _maxLevel : 4;

    std::vector<Vtr::Level *>      _levels;
    std::vector<Vtr::Refinement *> _refinements;

    std::vector<Index> _ptexIndices;
};

template <class T, class U>
inline void
TopologyRefiner::Interpolate(T const * src, U * dst) const {

    for (int level=1; level<=GetMaxLevel(); ++level) {

        Interpolate(level, src, dst);

        src = dst;
        dst += GetNumVertices(level);
    }
}

template <class T, class U>
inline void
TopologyRefiner::Interpolate(int level, T const & src, U & dst) const {

    assert(level>0 and level<=(int)_refinements.size());

    Vtr::Refinement const & refinement = getRefinement(level-1);

    switch (_subdivType) {
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
TopologyRefiner::interpolateChildVertsFromFaces(
    Vtr::Refinement const & refinement, T const & src, U & dst) const {

    if (refinement.getNumChildVerticesFromFaces() == 0) return;

    Sdc::Scheme<SCHEME> scheme(_subdivOptions);

    const Vtr::Level& parent = refinement.parent();

    float * fVertWeights = (float *)alloca(parent.getMaxValence()*sizeof(float));

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
TopologyRefiner::interpolateChildVertsFromEdges(
    Vtr::Refinement const & refinement, T const & src, U & dst) const {

    Sdc::Scheme<SCHEME> scheme(_subdivOptions);

    const Vtr::Level& parent = refinement.parent();
    const Vtr::Level& child  = refinement.child();

    Vtr::EdgeInterface eHood(parent);

    float   eVertWeights[2],
          * eFaceWeights = (float *)alloca(parent.getMaxEdgeFaces()*sizeof(float));

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
TopologyRefiner::interpolateChildVertsFromVerts(
    Vtr::Refinement const & refinement, T const & src, U & dst) const {

    Sdc::Scheme<SCHEME> scheme(_subdivOptions);

    const Vtr::Level& parent = refinement.parent();
    const Vtr::Level& child  = refinement.child();

    Vtr::VertexInterface vHood(parent, child);

    float * weightBuffer = (float *)alloca(2*parent.getMaxValence()*sizeof(float));

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
TopologyRefiner::InterpolateVarying(T const * src, U * dst) const {

    for (int level=1; level<=GetMaxLevel(); ++level) {

        InterpolateVarying(level, src, dst);

        src = dst;
        dst += GetNumVertices(level);
    }
}

template <class T, class U>
inline void
TopologyRefiner::InterpolateVarying(int level, T const & src, U & dst) const {

    assert(level>0 and level<=(int)_refinements.size());

    Vtr::Refinement const & refinement = getRefinement(level-1);

    varyingInterpolateChildVertsFromFaces(refinement, src, dst);
    varyingInterpolateChildVertsFromEdges(refinement, src, dst);
    varyingInterpolateChildVertsFromVerts(refinement, src, dst);
}

template <class T, class U>
inline void
TopologyRefiner::varyingInterpolateChildVertsFromFaces(
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
TopologyRefiner::varyingInterpolateChildVertsFromEdges(
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
TopologyRefiner::varyingInterpolateChildVertsFromVerts(
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
TopologyRefiner::InterpolateFaceVarying(T const * src, U * dst, int channel) const {

    for (int level=1; level<=GetMaxLevel(); ++level) {

        InterpolateFaceVarying(level, src, dst, channel);

        src = dst;
        dst += getLevel(level).getNumFVarValues();
    }
}

template <class T, class U>
inline void
TopologyRefiner::InterpolateFaceVarying(int level, T const & src, U & dst, int channel) const {

    assert(level>0 and level<=(int)_refinements.size());

    Vtr::Refinement const & refinement = getRefinement(level-1);

    switch (_subdivType) {
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
TopologyRefiner::faceVaryingInterpolateChildVertsFromFaces(
    Vtr::Refinement const & refinement, T const & src, U & dst, int channel) const {

    if (refinement.getNumChildVerticesFromFaces() == 0) return;

    Sdc::Scheme<SCHEME> scheme(_subdivOptions);

    const Vtr::Level& parentLevel = refinement.parent();
    const Vtr::Level& childLevel  = refinement.child();

    const Vtr::FVarLevel& parentFVar = *parentLevel._fvarChannels[channel];
    const Vtr::FVarLevel& childFVar  = *childLevel._fvarChannels[channel];

    float * fValueWeights = (float *)alloca(parentLevel.getMaxValence()*sizeof(float));

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
TopologyRefiner::faceVaryingInterpolateChildVertsFromEdges(
    Vtr::Refinement const & refinement, T const & src, U & dst, int channel) const {

    Sdc::Scheme<SCHEME> scheme(_subdivOptions);

    const Vtr::Level& parentLevel = refinement.parent();
    const Vtr::Level& childLevel  = refinement.child();

    const Vtr::FVarRefinement& refineFVar = *refinement._fvarChannels[channel];
    const Vtr::FVarLevel&      parentFVar = *parentLevel._fvarChannels[channel];
    const Vtr::FVarLevel&      childFVar  = *childLevel._fvarChannels[channel];

    //
    //  Allocate and intialize (if linearly interpolated) interpolation weights for
    //  the edge mask:
    //
    float   eVertWeights[2],
          * eFaceWeights = (float *)alloca(parentLevel.getMaxEdgeFaces()*sizeof(float));

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
TopologyRefiner::faceVaryingInterpolateChildVertsFromVerts(
    Vtr::Refinement const & refinement, T const & src, U & dst, int channel) const {

    Sdc::Scheme<SCHEME> scheme(_subdivOptions);

    const Vtr::Level& parentLevel = refinement.parent();
    const Vtr::Level& childLevel  = refinement.child();

    const Vtr::FVarRefinement& refineFVar = *refinement._fvarChannels[channel];
    const Vtr::FVarLevel&      parentFVar = *parentLevel._fvarChannels[channel];
    const Vtr::FVarLevel&      childFVar  = *childLevel._fvarChannels[channel];

    bool isLinearFVar = parentFVar._isLinear;

    float * weightBuffer = (float *)alloca(2*parentLevel.getMaxValence()*sizeof(float));

    Vtr::Index * vEdgeValues = (Vtr::Index *)alloca(parentLevel.getMaxValence()*sizeof(Vtr::Index));

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
TopologyRefiner::Limit(T const & src, U * dst) const {

    assert(GetMaxLevel() > 0);

    switch (_subdivType) {
    case Sdc::SCHEME_CATMARK:
        limit<Sdc::SCHEME_CATMARK>(src, dst);
        break;
    case Sdc::SCHEME_LOOP:
        limit<Sdc::SCHEME_LOOP>(src, dst);
        break;
    case Sdc::SCHEME_BILINEAR:
        limit<Sdc::SCHEME_BILINEAR>(src, dst);
        break;
    }
}

template <Sdc::SchemeType SCHEME, class T, class U>
inline void
TopologyRefiner::limit(T const & src, U * dst) const {

    //
    //  Work in progress...
    //      - does not support tangents yet (unclear how)
    //      - need to verify that each vertex is "limitable", i.e.:
    //          - is not semi-sharp, inf-sharp or non-manifold
    //          - is "complete" wrt its parent (if refinement is sparse)
    //      - copy (or weight by 1.0) src to dst when not "limitable"
    //      - currently requires one refinement to get rid of N-sided faces:
    //          - could limit regular vertices from level 0
    //
    Sdc::Scheme<SCHEME> scheme(_subdivOptions);

    Vtr::Level const & level = getLevel(GetMaxLevel());

    int maxWeightsPerMask = 1 + 2 * level.getMaxValence();

    float * weightBuffer = (float *)alloca(maxWeightsPerMask * sizeof(float));

    //  This is a bit obscure -- assign both parent and child as last level
    Vtr::VertexInterface vHood(level, level);

    for (int vert = 0; vert < level.getNumVertices(); ++vert) {
        ConstIndexArray vEdges = level.getVertexEdges(vert);

        float * vWeights = weightBuffer,
              * eWeights = vWeights + 1,
              * fWeights = eWeights + vEdges.size();

        Vtr::MaskInterface vMask(vWeights, eWeights, fWeights);

        //  This is a bit obscure -- child vertex index will be ignored here
        vHood.SetIndex(vert, vert);

        scheme.ComputeVertexLimitMask(vHood, vMask);

        //  Apply the weights to the vertex, the vertices opposite its incident
        //  edges, and the opposite vertices of its incident faces:
        //
        //  As with applying refinment masks to vertex data, in order to improve
        //  numerical precision, its better to apply smaller weights first, so
        //  begin with the face-weights followed by the edge-weights and the vertex
        //  weight last.

        dst[vert].Clear();
        if (vMask.GetNumFaceWeights() > 0) {
            assert(!vMask.AreFaceWeightsForFaceCenters());

            ConstIndexArray      vFaces = level.getVertexFaces(vert);
            ConstLocalIndexArray vInFace = level.getVertexFaceLocalIndices(vert);
            for (int i = 0; i < vFaces.size(); ++i) {
                ConstIndexArray fVerts = level.getFaceVertices(vFaces[i]);

                LocalIndex vOppInFace = (vInFace[i] + 2);
                if (vOppInFace >= fVerts.size()) vOppInFace -= (LocalIndex)fVerts.size();
                Index vertOppositeFace = level.getFaceVertices(vFaces[i])[vOppInFace];

                dst[vert].AddWithWeight(src[vertOppositeFace], fWeights[i]);
            }
        }
        if (vMask.GetNumEdgeWeights() > 0) {
            for (int i = 0; i < vEdges.size(); ++i) {
                ConstIndexArray eVerts = level.getEdgeVertices(vEdges[i]);
                Index vertOppositeEdge = (eVerts[0] == vert) ? eVerts[1] : eVerts[0];

                dst[vert].AddWithWeight(src[vertOppositeEdge], eWeights[i]);
            }
        }
        dst[vert].AddWithWeight(src[vert], vWeights[0]);
    }
}

template <class T, class U>
inline void
TopologyRefiner::LimitFaceVarying(T const & src, U * dst, int channel) const {

    assert(GetMaxLevel() > 0);

    switch (_subdivType) {
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
TopologyRefiner::faceVaryingLimit(T const & src, U * dst, int channel) const {

    Sdc::Scheme<SCHEME> scheme(_subdivOptions);

    Vtr::Level const &      level       = getLevel(GetMaxLevel());
    Vtr::FVarLevel const &  fvarChannel = *level._fvarChannels[channel];

    int maxWeightsPerMask = 1 + 2 * level.getMaxValence();

    float * weightBuffer = (float *)alloca(maxWeightsPerMask * sizeof(float));
    Index * indexBuffer  = (Index *)alloca(level.getMaxValence() * sizeof(Index));

    //  This is a bit obscure -- assign both parent and child as last level
    Vtr::VertexInterface vHood(level, level);

    for (int vert = 0; vert < level.getNumVertices(); ++vert) {

        ConstIndexArray vValues = fvarChannel.getVertexValues(vert);

        bool fvarVertMatchesVertex = fvarChannel.valueTopologyMatches(vValues[0]);
        if (fvarChannel._isLinear && fvarVertMatchesVertex) {
            Vtr::Index srcValueIndex = fvarChannel.getVertexValue(vert);
            Vtr::Index dstValueIndex = vValues[0];

            dst[dstValueIndex].Clear();
            dst[dstValueIndex].AddWithWeight(src[srcValueIndex], 1.0f);
        } else if (fvarVertMatchesVertex) {
            //
            //  Compute the limit mask based on vertex topology:
            //
            ConstIndexArray vEdges = level.getVertexEdges(vert);

            float * vWeights = weightBuffer,
                  * eWeights = vWeights + 1,
                  * fWeights = eWeights + vEdges.size();

            Vtr::MaskInterface vMask(vWeights, eWeights, fWeights);

            vHood.SetIndex(vert, vert);

            scheme.ComputeVertexLimitMask(vHood, vMask);

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
                Index * vEdgeValues = indexBuffer;
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

#endif /* FAR_TOPOLOGY_REFINER_H */
