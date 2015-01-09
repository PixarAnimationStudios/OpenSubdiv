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
#ifndef FAR_TOPOLOGY_REFINER_FACTORY_H
#define FAR_TOPOLOGY_REFINER_FACTORY_H

#include "../version.h"

#include "../far/topologyRefiner.h"
#include "../far/error.h"

#include <cassert>

#ifdef _MSC_VER
    #define snprintf _snprintf
#endif

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

//
//  TopologyRefinerFactoryBase:
//      This is an abstract base class for subclasses that are intended to construct
//  TopologyRefiner from external mesh representations.  These subclasses are
//  parameterized by the mesh type <class MESH>.  The base class provides all
//  implementation details related to assembly and validation that are independent
//  of the subclass' mesh type.
//
class TopologyRefinerFactoryBase {

public:

    /// \brief Descriptor for raw topology data, provided as a convenience for one
    /// particular factory, but not used by others.
    ///
    struct TopologyDescriptor {

        int           numVertices,
                      numFaces;

        int const   * numVertsPerFace;
        Index const * vertIndicesPerFace;

        int           numCreases;
        Index const * creaseVertexIndexPairs;
        float const * creaseWeights;

        int           numCorners;
        Index const * cornerVertexIndices;
        float const * cornerWeights;
        
        int           numHoles;
        Index const * holeIndices;

        //  Face-varying data channel -- value indices correspond to vertex indices,
        //  i.e. one for every vertex of every face:
        //
        struct FVarChannel {

            int         numValues;
            int const * valueIndices;

            FVarChannel() : numValues(0), valueIndices(0) { }
        };
        
        int                 numFVarChannels;
        FVarChannel const * fvarChannels;

        TopologyDescriptor();
    };

protected:

    //
    //  Protected methods invoked by the subclass template to verify and process each
    //  stage of construction implemented by the subclass:
    //
    typedef Vtr::Level::ValidationCallback TopologyCallback;

    static bool prepareComponentTopologySizing(TopologyRefiner& refiner);
    static bool prepareComponentTopologyAssignment(TopologyRefiner& refiner, bool fullValidation,
                                                   TopologyCallback callback, void const * callbackData);
    static bool prepareComponentTagsAndSharpness(TopologyRefiner& refiner);
    static bool prepareFaceVaryingChannels(TopologyRefiner& refiner);
};


//
//  TopologyRefinerFactory<MESH>:
//      The factory class template to convert and refine an instance of TopologyRefiner
//  from an arbitrary mesh class.  While a class template, the implementation is not
//  (cannot) be complete, so specialization of a few methods is required.
//      This template provides both the interface and high level assembly for the
//  construction of the TopologyRefiner instance.  The high level construction executes
//  a specific set of operations to convert the client's MESH into TopologyRefiner,
//  using methods independent of MESH from the base class and those specialized for
//  class MESH appropriately.
//
template <class MESH>
class TopologyRefinerFactory : public TopologyRefinerFactoryBase {

public:

    /// \brief Options related to the construction of each TopologyRefiner.
    ///
    struct Options {

        Options(Sdc::SchemeType sdcType = Sdc::SCHEME_CATMARK, Sdc::Options sdcOptions = Sdc::Options()) :
            schemeType(sdcType),
            schemeOptions(sdcOptions),
            validateFullTopology(false) { }

        Sdc::SchemeType schemeType;             ///< The subdivision scheme type identifier
        Sdc::Options    schemeOptions;          ///< The full set of options for the scheme,
                                                ///< e.g. boundary interpolation rules...
        unsigned int validateFullTopology : 1;  ///< Apply more extensive validation of
                                                ///< the constructed topology -- intended
                                                ///< for debugging.
    };

    /// \brief Instantiates TopologyRefiner from client-provided topological
    ///        representation.
    ///
    ///  If only the face-vertices topological relationships are specified
    ///  with this factory, edge relationships have to be inferred, which
    ///  requires additional processing. If the client topological rep can
    ///  provide this information, it is highly recommended to do so.
    ///
    /// @param mesh       Client's topological representation (or a converter)
    //
    /// @param options    Options controlling the creation of the TopologyRefiner
    ///
    /// return            A new instance of TopologyRefiner or NULL for failure
    ///
    static TopologyRefiner* Create(MESH const& mesh, Options options = Options());

protected:
    static bool populateBaseLevel(TopologyRefiner& refiner, MESH const& mesh, Options options);

    //
    //  Methods to be specialized that implement all details specific to class MESH required
    //  to convert MESH data to TopologyRefiner.  Note that some of these *must* be specialized
    //  in order to complete construction while some are optional.
    //
    //  There are two minimal construction requirements (to specify the size and content of
    //  all topology relations) and two optional (to specify feature tags and face-varying
    //  channels).
    //
    //  See comments in the generic stubs or the tutorials for more details on writing these.
    //
    //  Required:
    static bool resizeComponentTopology(TopologyRefiner& refiner, MESH const& mesh);
    static bool assignComponentTopology(TopologyRefiner& refiner, MESH const& mesh);

    //  Optional:
    static bool assignComponentTags(TopologyRefiner& refiner, MESH const& mesh);
    static bool assignFaceVaryingTopology(TopologyRefiner& refiner, MESH const& mesh);

    //  Optional miscellaneous specializations -- error reporting, etc.:
    typedef Vtr::Level::TopologyError TopologyError;

    static void reportInvalidTopology(TopologyError errCode, char const * msg, MESH const& mesh);
};


//
//  Generic implementations:
//
template <class MESH>
TopologyRefiner*
TopologyRefinerFactory<MESH>::Create(MESH const& mesh, Options options) {

    TopologyRefiner * refiner = new TopologyRefiner(options.schemeType, options.schemeOptions);

    if (not populateBaseLevel(*refiner, mesh, options)) {
        delete refiner;
        return 0;
    }

    // XXXX -- any state in the TopologyRefiner to update after the base level is complete?

    return refiner;
}

template <class MESH>
bool
TopologyRefinerFactory<MESH>::populateBaseLevel(TopologyRefiner& refiner, MESH const& mesh, Options options) {

    //
    //  Construction of a specialized topology refiner involves four steps, each of which
    //  involves a method specialized for MESH followed by one that takes an action in
    //  response to it or in preparation for the next step.
    //
    //  Both the specialized methods and those that follow them may find fault in the
    //  construction and trigger failure at any time:
    //

    //
    //  Sizing of the topology -- this is a required specialization for MESH.  This defines
    //  an inventory of all components and their relations that is used to allocate buffers
    //  to be efficiently populated in the subsequent topology assignment step.
    //
    if (not resizeComponentTopology(refiner, mesh)) return false;
    if (not prepareComponentTopologySizing(refiner)) return false;

    //
    //  Assignment of the topology -- this is a required specialization for MESH.  If edges
    //  are specified, all other topological relations are expected to be defined for them.
    //  Otherwise edges and remaining topology will be completed from the face-vertices:
    //
    bool             validate = options.validateFullTopology;
    TopologyCallback callback = reinterpret_cast<TopologyCallback>(reportInvalidTopology);
    void const *     userData = &mesh;
        
    if (not assignComponentTopology(refiner, mesh)) return false;
    if (not prepareComponentTopologyAssignment(refiner, validate, callback, userData)) return false;

    //
    //  User assigned and internal tagging of components -- an optional specialization for
    //  MESH.  Allows the specification of sharpness values, holes, etc.
    //
    if (not assignComponentTags(refiner, mesh)) return false;
    if (not prepareComponentTagsAndSharpness(refiner)) return false;

    //
    //  Defining channels of face-varying primvar data -- an optional specialization for MESH.
    //
    if (not assignFaceVaryingTopology(refiner, mesh)) return false;
    if (not prepareFaceVaryingChannels(refiner)) return false;

    return true;
}

// XXXX manuelk MSVC specializes these templated functions which creates duplicated symbols
#ifndef _MSC_VER

template <class MESH>
bool
TopologyRefinerFactory<MESH>::resizeComponentTopology(TopologyRefiner& /* refiner */, MESH const& /* mesh */) {

    assert("Missing specialization for TopologyRefinerFactory<MESH>::resizeComponentTopology()" == 0);

    //
    //  Sizing the topology tables:
    //      This method is for determining the sizes of the various topology tables (and other
    //  data) associated with the mesh.  Once completed, appropriate memory will be allocated
    //  and an additional method invoked to populate it accordingly.
    //
    //  The following methods should be called -- first those to specify the number of faces,
    //  edges and vertices in the mesh:
    //
    //      void TopologyRefiner::setBaseFaceCount(int count)
    //      void TopologyRefiner::setBaseEdgeCount(int count)
    //      void TopologyRefiner::setBaseVertexCount(int count)
    //
    //  and then for each face, edge and vertex, the number of its incident components:
    //
    //      void TopologyRefiner::setBaseFaceVertexCount(Index face, int count)
    //      void TopologyRefiner::setBaseEdgeFaceCount(  Index edge, int count)
    //      void TopologyRefiner::setBaseVertexFaceCount(Index vertex, int count)
    //      void TopologyRefiner::setBaseVertexEdgeCount(Index vertex, int count)
    //
    //  The count/size for a component type must be set before indices associated with that
    //  component type can be used.
    //
    //  Note that it is only necessary to size 4 of the 6 supported topological relations --
    //  the number of edge-vertices is fixed at two per edge, and the number of face-edges is
    //  the same as the number of face-vertices.
    //
    //  So a single pass through your mesh to gather up all of this sizing information will
    //  allow the Tables to be allocated appropriately once and avoid any dynamic resizing as
    //  it grows.
    //
    return false;
}

template <class MESH>
bool
TopologyRefinerFactory<MESH>::assignComponentTopology(TopologyRefiner& /* refiner */, MESH const& /* mesh */) {

    assert("Missing specialization for TopologyRefinerFactory<MESH>::assignComponentTopology()" == 0);

    //
    //  Assigning the topology tables:
    //      Once the topology tables have been allocated, the six required topological
    //  relations can be directly populated using the following methods:
    //
    //      IndexArray TopologyRefiner::setBaseFaceVertices(Index face)
    //      IndexArray TopologyRefiner::setBaseFaceEdges(Index face)
    //
    //      IndexArray TopologyRefiner::setBaseEdgeVertices(Index edge)
    //      IndexArray TopologyRefiner::setBaseEdgeFaces(Index edge)
    //
    //      IndexArray TopologyRefiner::setBaseVertexEdges(Index vertex)
    //      IndexArray TopologyRefiner::setBaseVertexFaces(Index vertex)
    //
    //  For the last two relations -- the faces and edges incident a vertex -- there are
    //  also "local indices" that must be specified (considering doing this internally),
    //  where the "local index" of each incident face or edge is the index of the vertex
    //  within that face or edge, and so ranging from 0-3 for incident quads and 0-1 for
    //  incident edges.  These are assigned through similarly retrieved arrays:
    //
    //      LocalIndexArray TopologyRefiner::setBaseVertexFaceLocalIndices(Index vertex)
    //      LocalIndexArray TopologyRefiner::setBaseVertexEdgeLocalIndices(Index vertex)
    //
    //  or, if the mesh is manifold, explicit assignment of these can be deferred and
    //  all will be determined via:
    //
    //      void TopologyRefiner::populateBaseLocalIndices()
    //
    //  All components are assumed to be locally manifold and ordering of components in
    //  the above relations is expected to be counter-clockwise.
    //
    //  For non-manifold components, no ordering/orientation of incident components is
    //  assumed or required, but be sure to explicitly tag such components (vertices and
    //  edges) as non-manifold:
    //
    //      void TopologyRefiner::setBaseEdgeNonManifold(Index edge, bool b);
    //
    //      void TopologyRefiner::setBaseVertexNonManifold(Index vertex, bool b);
    //
    //  Also consider using TopologyRefiner::ValidateTopology() when debugging to ensure
    //  that topolology has been completely and correctly specified.
    //
    return false;
}

template <class MESH>
bool
TopologyRefinerFactory<MESH>::assignFaceVaryingTopology(TopologyRefiner& /* refiner */, MESH const& /* mesh */) {

    //
    //  Optional assigning face-varying topology tables:
    //
    //  Create independent face-varying primitive variable channels:
    //      int TopologyRefiner::createBaseFVarChannel(int numValues)
    //
    //  For each channel, populate the face-vertex values:
    //      IndexArray TopologyRefiner::setBaseFVarFaceValues(Index face, int channel = 0)
    //
    return true;
}

template <class MESH>
bool
TopologyRefinerFactory<MESH>::assignComponentTags(TopologyRefiner& /* refiner */, MESH const& /* mesh */) {

    //
    //  Optional tagging:
    //      This is where any additional feature tags -- sharpness, holes, etc. -- can be
    //  specified using:
    //
    //      void TopologyRefiner::setBaseEdgeSharpness(Index edge, float sharpness)
    //      void TopologyRefiner::setBaseVertexSharpness(Index vertex, float sharpness)
    //
    //      void TopologyRefiner::setBaseFaceHole(Index face, bool hole)
    //
    return true;
}

template <class MESH>
void
TopologyRefinerFactory<MESH>::reportInvalidTopology(
    TopologyError /* errCode */, char const * /* msg */, MESH const& /* mesh */) {

    //
    //  Optional topology validation error reporting:
    //      This method is called whenever the factory encounters topology validation
    //  errors. By default, nothing is reported
    //
}

#endif

//
// Specialization for raw topology data
//
template <>
bool
TopologyRefinerFactory<TopologyRefinerFactoryBase::TopologyDescriptor>::resizeComponentTopology(
    TopologyRefiner & refiner, TopologyDescriptor const & desc);

template <>
bool
TopologyRefinerFactory<TopologyRefinerFactoryBase::TopologyDescriptor>::assignComponentTopology(
    TopologyRefiner & refiner, TopologyDescriptor const & desc);

template <>
bool
TopologyRefinerFactory<TopologyRefinerFactoryBase::TopologyDescriptor>::assignComponentTags(
    TopologyRefiner & refiner, TopologyDescriptor const & desc);

template <>
bool
TopologyRefinerFactory<TopologyRefinerFactoryBase::TopologyDescriptor>::assignFaceVaryingTopology(
    TopologyRefiner & refiner, TopologyDescriptor const & desc);

template <>
void
TopologyRefinerFactory<TopologyRefinerFactoryBase::TopologyDescriptor>::reportInvalidTopology(
    TopologyError errCode, char const * msg, TopologyDescriptor const& /* mesh */);

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;
} // end namespace OpenSubdiv

#endif /* FAR_TOPOLOGY_REFINER_FACTORY_H */
