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
#include "../sdc/options.h"
#include "../sdc/crease.h"
#include "../vtr/level.h"
#include "../far/topologyRefiner.h"
#include "../far/topologyRefinerFactory.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

//
//  Methods for the Factory base class -- general enough to warrant including in
//  the base class rather than the subclass template (and so replicated for each
//  usage)
//
//
void
TopologyRefinerFactoryBase::validateComponentTopologySizing(TopologyRefiner& refiner) {

    Vtr::Level& baseLevel = refiner.getBaseLevel();

    int vCount = baseLevel.getNumVertices();
    int eCount = baseLevel.getNumEdges();
    int fCount = baseLevel.getNumFaces();

    assert((vCount > 0) && (fCount > 0));

    //
    //  This still needs a little work -- currently we are assuming all counts and offsets
    //  have been assigned, but eventually only the counts will be assigined (in arbitrary
    //  order) and we will need to accumulate the offsets to get the total sizes.  That
    //  will require new methods on Vtr::Level -- we do not want direct member access here.
    //
    int fVertCount = 0;
    for (int i = 0; i < fCount; ++i) {
        fVertCount += baseLevel.getNumFaceVertices(i);
    }
    baseLevel.resizeFaceVertices(fVertCount);
    assert(baseLevel.getNumFaceVerticesTotal() > 0);

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
}

void
TopologyRefinerFactoryBase::validateVertexComponentTopologyAssignment(TopologyRefiner& refiner) {

    Vtr::Level& baseLevel = refiner.getBaseLevel();

    //
    //  In future we may want the ability to complete aspects of the topology that are incovenient
    //  for clients to specify, e.g. the local indices associated with some relations, orienting
    //  the vertex relations, etc.  For the near term we'll be assuming only face-vertices have
    //  been specified and the absence of edges will trigger the construction of everything else:
    //
    bool completeMissingTopology = (baseLevel.getNumEdges() == 0);
    if (completeMissingTopology) {
        //  Need to invoke some Vtr::Level method to "fill in" the missing topology...
        baseLevel.completeTopologyFromFaceVertices();
    }

    bool applyValidation = false;
    if (applyValidation) {
        if (!baseLevel.validateTopology()) {
            printf("Invalid topology detected in TopologyRefinerFactory (%s)\n",
                completeMissingTopology ? "partially specified and completed" : "fully specified");
            //baseLevel.print();
            assert(false);
        }
    }
}

void
TopologyRefinerFactoryBase::validateFaceVaryingComponentTopologyAssignment(TopologyRefiner& refiner) {

    for (int channel=0; channel<refiner.GetNumFVarChannels(); ++channel) {
        refiner.completeFVarChannelTopology(channel);
    }
}

//
//  This method combines the initialization of component tags with the sharpening of edges and
//  vertices according to the given boundary interpolation rule in the Options.  Since both
//  involve traversing the edge and vertex lists and noting the presence of boundaries -- best
//  to do both at once...
//
void
TopologyRefinerFactoryBase::applyComponentTagsAndBoundarySharpness(TopologyRefiner& refiner) {

    Vtr::Level&  baseLevel = refiner.getBaseLevel();

    assert((int)baseLevel._edgeTags.size() == baseLevel.getNumEdges());
    assert((int)baseLevel._vertTags.size() == baseLevel.getNumVertices());
    assert((int)baseLevel._faceTags.size() == baseLevel.getNumFaces());

    Sdc::Options options = refiner.GetSchemeOptions();
    Sdc::Crease  creasing(options);

    bool sharpenCornerVerts    = (options.GetVVarBoundaryInterpolation() == Sdc::Options::VVAR_BOUNDARY_EDGE_AND_CORNER);
    bool sharpenNonManFeatures = (options.GetNonManifoldInterpolation() == Sdc::Options::NON_MANIFOLD_SHARP);

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
        assert(refiner.GetSchemeType() == Sdc::TYPE_CATMARK);

        vTag._boundary = (vFaces.size() < vEdges.size());
        if (isCorner) {
            vTag._xordinary = !sharpenCornerVerts;
        } else if (vTag._boundary) {
            vTag._xordinary = (vFaces.size() != 2);
        } else {
            vTag._xordinary = (vFaces.size() != 4);
        }
    }

    //
    //  Anything more to be done with Face tags? (eventually when processing edits perhaps)
    //
    //  for (Vtr::Index fIndex = 0; fIndex < baseLevel.getNumFaces(); ++fIndex) {
    //  }
}

//
// Specialization for raw topology data
//
template <>
void
TopologyRefinerFactory<TopologyRefinerFactoryBase::TopologyDescriptor>::resizeComponentTopology(
    TopologyRefiner & refiner, TopologyDescriptor const & desc) {

    refiner.setNumBaseVertices(desc.numVertices);
    refiner.setNumBaseFaces(desc.numFaces);

    for (int face=0; face<desc.numFaces; ++face) {

        refiner.setNumBaseFaceVertices(face, desc.vertsPerFace[face]);
    }
}

template <>
void
TopologyRefinerFactory<TopologyRefinerFactoryBase::TopologyDescriptor>::assignComponentTopology(
    TopologyRefiner & refiner, TopologyDescriptor const & desc) {

    for (int face=0, idx=0; face<desc.numFaces; ++face) {

        IndexArray dstFaceVerts = refiner.setBaseFaceVertices(face);

        for (int vert=0; vert<dstFaceVerts.size(); ++vert) {

            dstFaceVerts[vert] = desc.vertIndices[idx++];
        }
    }
}

template <>
void
TopologyRefinerFactory<TopologyRefinerFactoryBase::TopologyDescriptor>::assignFaceVaryingTopology(
    TopologyRefiner & refiner, TopologyDescriptor const & desc) {

    if (desc.numFVarChannels>0) {

        for (int channel=0; channel<desc.numFVarChannels; ++channel) {

            int        channelSize    = desc.fvarChannels[channel].numValues;
            int const* channelIndices = desc.fvarChannels[channel].valueIndices;

#if defined(DEBUG) or defined(_DEBUG)
            int channelIndex = refiner.createFVarChannel(channelSize);
            assert(channelIndex == channel);
#else
            refiner.createFVarChannel(channelSize);
#endif
            for (int face=0, idx=0; face<desc.numFaces; ++face) {

                IndexArray dstFaceValues = refiner.getBaseFVarFaceValues(face, channel);

                for (int vert=0; vert<dstFaceValues.size(); ++vert) {

                    dstFaceValues[vert] = channelIndices[idx++];
                }
            }
        }
    }
}

template <>
void
TopologyRefinerFactory<TopologyRefinerFactoryBase::TopologyDescriptor>::assignComponentTags(
    TopologyRefiner & refiner, TopologyDescriptor const & desc) {


    if ((desc.numCreases>0) and desc.creaseVertexIndexPairs and desc.creaseWeights) {

        int const * vertIndexPairs = desc.creaseVertexIndexPairs;
        for (int edge=0; edge<desc.numCreases; ++edge, vertIndexPairs+=2) {

            Index idx = refiner.FindEdge(0, vertIndexPairs[0], vertIndexPairs[1]);

            if (idx!=Vtr::INDEX_INVALID) {
                refiner.baseEdgeSharpness(idx) = desc.creaseWeights[edge];
            } else {
                // XXXX report error !
            }
        }
    }

    if ((desc.numCorners>0) and desc.cornerVertexIndices and desc.cornerWeights) {

        for (int vert=0; vert<desc.numCorners; ++vert) {

            int idx = desc.cornerVertexIndices[vert];

            if (idx < refiner.GetNumVertices(0)) {
                refiner.baseVertexSharpness(idx) = desc.cornerWeights[vert];
            } else {
                // XXXX report error !
            }
        }
    }

}

TopologyRefinerFactoryBase::TopologyDescriptor::TopologyDescriptor() :
    numVertices(0), numFaces(0), vertsPerFace(0), vertIndices(0),
        numCreases(0), creaseVertexIndexPairs(0), creaseWeights(0),
            numCorners(0), cornerVertexIndices(0), cornerWeights(0),
                numFVarChannels(0), fvarChannels(0) {
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
