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

#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

//
//  TopologyRefinerFactoryBase:
//      This is an abstract base class for subclasses that are intended to construct
//  TopologyRefiner from a mesh class that defines the subclass.  The subclasses are
//  parameterized by the mesh type <class MESH>.  The base class provides all
//  implementation details related to assembly and validation that are independent
//  of the subclass' mesh type.
//
//  Its still unclear where the division of functionality lies between construction
//  of the Factory and creation of an instance of the Tables.  We definitely do not
//  want data generation in the construction to require duplication or copying in
//  the creation.  Overloading Create() to "close" the tables (copying the base level
//  and other data, but refining differently) are also possibilities.
//
//  The subdiv type/options are specified on construction of the factory and are passed
//  on to each instance of TopologyRefiner that it creates.  They can be modified as
//  there is nothing in the Factory tied to these properties.  Consider overloading
//  the Create() method (defined by subclasses) to vary these if greater flexibility
//  per instance is desired.
//
class TopologyRefinerFactoryBase {

public:

    /// \brief Descriptor for raw topology data
    ///
    struct TopologyDescriptor {

        int           numVertices,
                      numFaces;

        int const   * vertsPerFace,
                    * vertIndices;

        int           numCreases;
        int const   * creaseVertexIndexPairs;
        float const * creaseWeights;

        int           numCorners;
        int const   * cornerVertexIndices;
        float const * cornerWeights;

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

    static void validateComponentTopologySizing(TopologyRefiner& refiner);
    static void validateVertexComponentTopologyAssignment(TopologyRefiner& refiner);
    static void validateFaceVaryingComponentTopologyAssignment(TopologyRefiner& refiner);

    static void applyComponentTagsAndBoundarySharpness(TopologyRefiner& refiner);
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

    /// \brief Constructor
    TopologyRefinerFactory() : TopologyRefinerFactoryBase() { }

    /// \brief Destructor
    ~TopologyRefinerFactory() { }

    /// \brief Instantiates TopologyRefiner from client-provided topological
    ///        representation.
    ///
    ///  If only the face-vertices topological relationships are specified
    ///  with this factory, edge relationships have to be inferred, which
    ///  requires additional processing. If the client topological rep can
    ///  provide this information, it is highly recommended to do so.
    ///
    /// @param type          The subdivision scheme
    ///
    /// @param options       Subdivion options (boundary interpolation rules...)
    ///
    /// @param mesh          Client topological representation (or a converter)
    ///
    /// return               An instance of TopologyRefiner or NULL for failure
    ///
    static TopologyRefiner* Create(Sdc::Type type, Sdc::Options options, MESH const& mesh);

protected:
    //
    //  Methods to be specialized that implement all details specific to class MESH required
    //  to convert MESH data to TopologyRefiner.  Note that some of these *must* be specialized
    //  in order to complete construction.
    //
    //  There are two minimal construction requirements and one optional.
    //
    //  See the comments in the generic stubs for details on how to write these.
    //
    //  Required:
    static void resizeComponentTopology(TopologyRefiner& refiner, MESH const& mesh);
    static void assignComponentTopology(TopologyRefiner& refiner, MESH const& mesh);

    //  Optional:
    static void assignFaceVaryingTopology(TopologyRefiner& refiner, MESH const& mesh);
    static void assignComponentTags(TopologyRefiner& refiner, MESH const& mesh);

protected:

    //  Other protected details -- not to be specialized:
    static void populateBaseLevel(TopologyRefiner& refiner, MESH const& mesh);
};


//
//  Generic implementations:
//
template <class MESH>
TopologyRefiner*
TopologyRefinerFactory<MESH>::Create(Sdc::Type type, Sdc::Options options, MESH const& mesh) {

    TopologyRefiner *refiner = new TopologyRefiner(type, options);

    populateBaseLevel(*refiner, mesh);

    return refiner;
}

template <class MESH>
void
TopologyRefinerFactory<MESH>::populateBaseLevel(TopologyRefiner& refiner, MESH const& mesh) {

    //
    //  The following three methods may end up virtual:
    //      - resize the component counts and relation counts for individual components:
    //      - assign the topological relations for all components:
    //      - assign any sharpness values, hole tags, etc:
    //  Note that we can do some sanity checking (independent of the type MESH) between these
    //  to ensure that a client has done what is necessary at each stage.
    //

    //  Required specialization for MESH:
    resizeComponentTopology(refiner, mesh);

    validateComponentTopologySizing(refiner);

    //  Required specialization for MESH:
    assignComponentTopology(refiner, mesh);
    validateVertexComponentTopologyAssignment(refiner);

    //  Optional specialization for MESH:
    assignComponentTags(refiner, mesh);

    //  Finalize the translation of the mesh after its full specification above:
    applyComponentTagsAndBoundarySharpness(refiner);

    //  Optional specialization for MESH:
    assignFaceVaryingTopology(refiner, mesh);
    validateFaceVaryingComponentTopologyAssignment(refiner);
}

// XXXX manuelk MSVC specializes these templated functions which creates duplicated symbols
#ifndef _MSC_VER

template <class MESH>
void
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
}

template <class MESH>
void
TopologyRefinerFactory<MESH>::assignComponentTopology(TopologyRefiner& /* refiner */, MESH const& /* mesh */) {

    assert("Missing specialization for TopologyRefinerFactory<MESH>::assignComponentTopology()" == 0);

    //
    //  Assigning the topology tables:
    //      Once the topology tables have been allocated, the six required topological
    //  relations can be directly populated using the following methods:
    //
    //      void IndexArray TopologyRefiner::baseFaceVertices(Index face)
    //      void IndexArray TopologyRefiner::baseFaceEdges(Index face)
    //
    //      void IndexArray TopologyRefiner::baseEdgeVertices(Index edge)
    //      void IndexArray TopologyRefiner::baseEdgeFaces(Index edge)
    //
    //      void IndexArray TopologyRefiner::baseVertexEdges(Index vertex)
    //      void IndexArray TopologyRefiner::baseVertexFaces(Index vertex)
    //
    //  For the last two relations -- the faces and edges incident a vertex -- there are
    //  also "local indices" that must be specified (considering doing this internally),
    //  where the "local index" of each incident face or edge is the index of the vertex
    //  within that face or edge, and so ranging from 0-3 for incident quads and 0-1 for
    //  incident edges.  These are assigned through similarly retrieved arrays:
    //
    //      LocalIndexArray TopologyRefiner::baseVertexFaceLocalIndices(Index vertex)
    //      LocalIndexArray TopologyRefiner::baseVertexEdgeLocalIndices(Index vertex)
    //
    //  As noted, we are considering determining these internally to avoid this complexity,
    //  but that will require iteration through the sets of vertex-faces and edges to find
    //  the location of the vertex within each.  If that is known at the time the incident
    //  componets are assigned, they we can avoid that separate pass.
    //
    //  We also need to tag vertices as manifold or not here.  Failure to do so explicitly
    //  will require the factory analyze the local neighborhood of each component, which
    //  is costly and often unnecessary.
    //
}

template <class MESH>
void
TopologyRefinerFactory<MESH>::assignFaceVaryingTopology(TopologyRefiner& /* refiner */, MESH const& /* mesh */) {

    //
    //  Optional assigning face-varying topology tables:
    //
    //  Create independent face-varying primitive variable channels:
    //      int TopologyRefiner::createFVarChannel(int numValues)
    //
    //  For each channel, populate the face-vertex values:
    //      IndexArray TopologyRefiner::getBaseFVarFaceValues(Index face, int channel = 0)
    //
}

template <class MESH>
void
TopologyRefinerFactory<MESH>::assignComponentTags(TopologyRefiner& /* refiner */, MESH const& /* mesh */) {

    //
    //  Optional tagging:
    //      This is where any additional feature tags -- sharpness, holes, etc. -- can be
    //  specified.  For now, this is limited to sharpness using the following:
    //
    //      float& TopologyRefiner::baseEdgeSharpness(Index edge)
    //      float& TopologyRefiner::baseVertexSharpness(Index vertex)
    //
    //  which can be used on the LHS of assignments.
    //
    //  Tagging holes will become available in the near future as sets of bitfields for
    //  each component type are introduced and propogated through the refinement hierarchy.
    //
}

#endif

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;
} // end namespace OpenSubdiv

#endif /* FAR_TOPOLOGY_REFINER_FACTORY_H */
