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
#include "../far/topologyRefinerFactory.h"
#include "../far/topologyRefiner.h"
#include "../vtr/level.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

//
//  Methods for the Factory base class -- general enough to warrant including in
//  the base class rather than the subclass template (and so replicated for each
//  usage)
//
//
bool
TopologyRefinerFactoryBase::prepareComponentTopologySizing(TopologyRefiner& refiner) {

    Vtr::Level& baseLevel = refiner.getLevel(0);

    //
    //  At minimum we require face-vertices (the total count of which can be determined
    //  from the offsets accumulated during sizing pass) and we need to resize members
    //  related to them to be populated during assignment:
    //
    int vCount = baseLevel.getNumVertices();
    int fCount = baseLevel.getNumFaces();

    assert((vCount > 0) && (fCount > 0));

    int fVertCount = baseLevel.getNumFaceVertices(fCount - 1) +
                     baseLevel.getOffsetOfFaceVertices(fCount - 1);

    baseLevel.resizeFaceVertices(fVertCount);
    assert(baseLevel.getNumFaceVerticesTotal() > 0);

    //
    //  If edges were sized, all other topological relations must be sized with it, in
    //  which case we allocate those members to be populated.  Otherwise, sizing of the
    //  other topology members is deferred until the face-vertices are assigned and the
    //  resulting relationships determined:
    //
    int eCount = baseLevel.getNumEdges();

    if (eCount > 0) {
        baseLevel.resizeFaceEdges(baseLevel.getNumFaceVerticesTotal());
        baseLevel.resizeEdgeVertices();
        baseLevel.resizeEdgeFaces(  baseLevel.getNumEdgeFaces(eCount-1)   + baseLevel.getOffsetOfEdgeFaces(eCount-1));
        baseLevel.resizeVertexFaces(baseLevel.getNumVertexFaces(vCount-1) + baseLevel.getOffsetOfVertexFaces(vCount-1));
        baseLevel.resizeVertexEdges(baseLevel.getNumVertexEdges(vCount-1) + baseLevel.getOffsetOfVertexEdges(vCount-1));

        assert(baseLevel.getNumFaceEdgesTotal() > 0);
        assert(baseLevel.getNumEdgeVerticesTotal() > 0);
        assert(baseLevel.getNumEdgeFacesTotal() > 0);
        assert(baseLevel.getNumVertexFacesTotal() > 0);
        assert(baseLevel.getNumVertexEdgesTotal() > 0);
    }
    return true;
}

bool
TopologyRefinerFactoryBase::prepareComponentTopologyAssignment(TopologyRefiner& refiner, bool fullValidation,
                                                               TopologyCallback callback, void const * callbackData) {

    Vtr::Level& baseLevel = refiner.getLevel(0);

    bool completeMissingTopology = (baseLevel.getNumEdges() == 0);
    if (completeMissingTopology) {
        baseLevel.completeTopologyFromFaceVertices();
    }

    bool valid = true;
    if (fullValidation) {
        valid = baseLevel.validateTopology(callback, callbackData);
        if (not valid) {
            char msg[1024];
            snprintf(msg, 1024, "Invalid topology detected in TopologyRefinerFactory (%s)\n",
                completeMissingTopology ? "partially specified and completed" : "fully specified");
            Warning(msg);
        }
    }
    return valid;
}

bool
TopologyRefinerFactoryBase::prepareComponentTagsAndSharpness(TopologyRefiner& refiner) {

    //
    //  This method combines the initialization of internal component tags with the sharpening
    //  of edges and vertices according to the given boundary interpolation rule in the Options.
    //  Since both involve traversing the edge and vertex lists and noting the presence of
    //  boundaries -- best to do both at once...
    //
    Vtr::Level&  baseLevel = refiner.getLevel(0);

    assert((int)baseLevel._edgeTags.size() == baseLevel.getNumEdges());
    assert((int)baseLevel._vertTags.size() == baseLevel.getNumVertices());
    assert((int)baseLevel._faceTags.size() == baseLevel.getNumFaces());

    Sdc::Options options = refiner.GetSchemeOptions();
    Sdc::Crease  creasing(options);

    bool sharpenCornerVerts    = (options.GetVtxBoundaryInterpolation() == Sdc::Options::VTX_BOUNDARY_EDGE_AND_CORNER);
    bool sharpenNonManFeatures = true; //(options.GetNonManifoldInterpolation() == Sdc::Options::NON_MANIFOLD_SHARP);

    //
    //  Process the Edge tags first, as Vertex tags (notably the Rule) are dependent on
    //  properties of their incident edges.
    //
    for (Vtr::Index eIndex = 0; eIndex < baseLevel.getNumEdges(); ++eIndex) {
        Vtr::Level::ETag& eTag       = baseLevel._edgeTags[eIndex];
        float&          eSharpness = baseLevel._edgeSharpness[eIndex];

        eTag._boundary = (baseLevel._edgeFaceCountsAndOffsets[eIndex*2 + 0] < 2);
        if (eTag._boundary || (eTag._nonManifold && sharpenNonManFeatures)) {
            eSharpness = Sdc::Crease::SHARPNESS_INFINITE;
        }
        eTag._infSharp  = Sdc::Crease::IsInfinite(eSharpness);
        eTag._semiSharp = Sdc::Crease::IsSharp(eSharpness) && !eTag._infSharp;
    }

    //
    //  Process the Vertex tags now -- for some tags (semi-sharp and its rule) we need
    //  to inspect all incident edges:
    //
    int schemeRegularBoundaryValence = 2;
    int schemeRegularInteriorValence = 4;
    if (refiner.GetSchemeType() == Sdc::SCHEME_LOOP) {
        schemeRegularBoundaryValence = 3;
        schemeRegularInteriorValence = 6;
    }

    for (Vtr::Index vIndex = 0; vIndex < baseLevel.getNumVertices(); ++vIndex) {
        Vtr::Level::VTag& vTag       = baseLevel._vertTags[vIndex];
        float&          vSharpness = baseLevel._vertSharpness[vIndex];

        Vtr::IndexArray const vEdges = baseLevel.getVertexEdges(vIndex);
        Vtr::IndexArray const vFaces = baseLevel.getVertexFaces(vIndex);

        //
        //  Take inventory of properties of incident edges that affect this vertex:
        //
        int infSharpEdgeCount    = 0;
        int semiSharpEdgeCount   = 0;
        int nonManifoldEdgeCount = 0;
        for (int i = 0; i < vEdges.size(); ++i) {
            Vtr::Level::ETag const& eTag = baseLevel._edgeTags[vEdges[i]];

            infSharpEdgeCount    += eTag._infSharp;
            semiSharpEdgeCount   += eTag._semiSharp;
            nonManifoldEdgeCount += eTag._nonManifold;
        }
        int sharpEdgeCount = infSharpEdgeCount + semiSharpEdgeCount;

        //
        //  Sharpen the vertex before using it in conjunction with incident edge
        //  properties to determine the semi-sharp tag and rule:
        //
        bool isCorner = (vFaces.size() == 1) && (vEdges.size() == 2);
        if (isCorner && sharpenCornerVerts) {
            vSharpness = Sdc::Crease::SHARPNESS_INFINITE;
        } else if (vTag._nonManifold && sharpenNonManFeatures) {
            //  Don't sharpen the vertex if a non-manifold crease:
            if (nonManifoldEdgeCount != 2) {
                vSharpness = Sdc::Crease::SHARPNESS_INFINITE;
            }
        }

        vTag._infSharp = Sdc::Crease::IsInfinite(vSharpness);

        vTag._semiSharp = Sdc::Crease::IsSemiSharp(vSharpness) || (semiSharpEdgeCount > 0);

        vTag._rule = (Vtr::Level::VTag::VTagSize)creasing.DetermineVertexVertexRule(vSharpness, sharpEdgeCount);

        //
        //  Assign topological tags -- note that the "xordinary" (or conversely a "regular")
        //  tag is still being considered, but regardless, it depends on the Sdc::Scheme...
        //
        vTag._boundary = (vFaces.size() < vEdges.size());
        if (isCorner) {
            vTag._xordinary = !sharpenCornerVerts;
        } else if (vTag._boundary) {
            vTag._xordinary = (vFaces.size() != schemeRegularBoundaryValence);
        } else {
            vTag._xordinary = (vFaces.size() != schemeRegularInteriorValence);
        }
        vTag._incomplete = 0;
    }

    return true;
}

bool
TopologyRefinerFactoryBase::prepareFaceVaryingChannels(TopologyRefiner& refiner) {

    Vtr::Level& baseLevel = refiner.getLevel(0);

    for (int channel=0; channel<refiner.GetNumFVarChannels(); ++channel) {
        baseLevel.completeFVarChannelTopology(channel);
    }
    return true;
}


//
// Specialization for raw topology data
//
template <>
bool
TopologyRefinerFactory<TopologyRefinerFactoryBase::TopologyDescriptor>::resizeComponentTopology(
    TopologyRefiner & refiner, TopologyDescriptor const & desc) {

    refiner.setNumBaseVertices(desc.numVertices);
    refiner.setNumBaseFaces(desc.numFaces);

    for (int face=0; face<desc.numFaces; ++face) {

        refiner.setNumBaseFaceVertices(face, desc.numVertsPerFace[face]);
    }
    return true;
}

template <>
bool
TopologyRefinerFactory<TopologyRefinerFactoryBase::TopologyDescriptor>::assignComponentTopology(
    TopologyRefiner & refiner, TopologyDescriptor const & desc) {

    for (int face=0, idx=0; face<desc.numFaces; ++face) {

        IndexArray dstFaceVerts = refiner.setBaseFaceVertices(face);

        for (int vert=0; vert<dstFaceVerts.size(); ++vert) {

            dstFaceVerts[vert] = desc.vertIndicesPerFace[idx++];
        }
    }
    return true;
}

template <>
bool
TopologyRefinerFactory<TopologyRefinerFactoryBase::TopologyDescriptor>::assignComponentTags(
    TopologyRefiner & refiner, TopologyDescriptor const & desc) {


    if ((desc.numCreases>0) and desc.creaseVertexIndexPairs and desc.creaseWeights) {

        int const * vertIndexPairs = desc.creaseVertexIndexPairs;
        for (int edge=0; edge<desc.numCreases; ++edge, vertIndexPairs+=2) {

            Index idx = refiner.FindEdge(0, vertIndexPairs[0], vertIndexPairs[1]);

            if (idx!=Vtr::INDEX_INVALID) {
                refiner.setBaseEdgeSharpness(idx, desc.creaseWeights[edge]);
            } else {
                char msg[1024];
                snprintf(msg, 1024, "Edge %d specified to be sharp does not exist (%d, %d)",
                    edge, vertIndexPairs[0], vertIndexPairs[1]);
                reportInvalidTopology(Vtr::Level::TOPOLOGY_INVALID_CREASE_EDGE, msg, desc);
            }
        }
    }

    if ((desc.numCorners>0) and desc.cornerVertexIndices and desc.cornerWeights) {

        for (int vert=0; vert<desc.numCorners; ++vert) {

            int idx = desc.cornerVertexIndices[vert];

            if (idx > 0 and idx < refiner.GetNumVertices(0)) {
                refiner.setBaseVertexSharpness(idx, desc.cornerWeights[vert]);
            } else {
                char msg[1024];
                snprintf(msg, 1024, "Vertex %d specified to be sharp does not exist", idx);
                reportInvalidTopology(Vtr::Level::TOPOLOGY_INVALID_CREASE_VERT, msg, desc);
            }
        }
    }
    if (desc.numHoles>0) {
        for (int i=0; i<desc.numHoles; ++i) {
            refiner.setBaseFaceHole(desc.holeIndices[i], true);
        }
    }
    return true;
}

template <>
bool
TopologyRefinerFactory<TopologyRefinerFactoryBase::TopologyDescriptor>::assignFaceVaryingTopology(
    TopologyRefiner & refiner, TopologyDescriptor const & desc) {

    if (desc.numFVarChannels>0) {

        for (int channel=0; channel<desc.numFVarChannels; ++channel) {

            int        channelSize    = desc.fvarChannels[channel].numValues;
            int const* channelIndices = desc.fvarChannels[channel].valueIndices;

#if defined(DEBUG) or defined(_DEBUG)
            int channelIndex = refiner.createBaseFVarChannel(channelSize);
            assert(channelIndex == channel);
#else
            refiner.createBaseFVarChannel(channelSize);
#endif
            for (int face=0, idx=0; face<desc.numFaces; ++face) {

                IndexArray dstFaceValues = refiner.setBaseFVarFaceValues(face, channel);

                for (int vert=0; vert<dstFaceValues.size(); ++vert) {

                    dstFaceValues[vert] = channelIndices[idx++];
                }
            }
        }
    }
    return true;
}

template <>
void
TopologyRefinerFactory<TopologyRefinerFactoryBase::TopologyDescriptor>::reportInvalidTopology(
    TopologyError /* errCode */, char const * msg, TopologyDescriptor const& /* mesh */) {
    Warning(msg);
}

TopologyRefinerFactoryBase::TopologyDescriptor::TopologyDescriptor() {
    memset(this, 0, sizeof(TopologyDescriptor));
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
