//
//   Copyright 2021 Pixar
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

#include "../../../regression/common/far_utils.h"

#include <opensubdiv/far/topologyRefiner.h>
#include <opensubdiv/far/topologyDescriptor.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

//  Utilities local to this tutorial:
namespace tutorial {

using namespace OpenSubdiv;

//
//  Create a TopologyRefiner from default geometry:
//
Far::TopologyRefiner *
dfltTopologyRefiner(std::vector<float> & posVector,
                    std::vector<float> & uvVector) {

    //
    //  Default topology and positions for a cube:
    //
    int dfltNumFaces = 6;
    int dfltNumVerts = 8;
    int dfltNumUVs   = 16;

    int dfltFaceSizes[6] = { 4, 4, 4, 4, 4, 4 };

    int dfltFaceVerts[24] = { 0, 1, 3, 2,
                              2, 3, 5, 4,
                              4, 5, 7, 6,
                              6, 7, 1, 0,
                              1, 7, 5, 3,
                              6, 0, 2, 4 };

    float dfltPositions[8][3] = {{ -0.5f, -0.5f,  0.5f },
                                 {  0.5f, -0.5f,  0.5f },
                                 { -0.5f,  0.5f,  0.5f },
                                 {  0.5f,  0.5f,  0.5f },
                                 { -0.5f,  0.5f, -0.5f },
                                 {  0.5f,  0.5f, -0.5f },
                                 { -0.5f, -0.5f, -0.5f },
                                 {  0.5f, -0.5f, -0.5f }};

    int dfltFaceFVars[24] = {  9, 10, 14, 13,
                               4,  0,  1,  5,
                               5,  1,  2,  6,
                               6,  2,  3,  7,
                              10, 11, 15, 14,
                               8,  9, 13, 12 };

    float dfltUVs[16][2] = {{ 0.05f, 0.05f },
                            { 0.35f, 0.15f },
                            { 0.65f, 0.15f },
                            { 0.95f, 0.05f },
                            { 0.05f, 0.35f },
                            { 0.35f, 0.45f },
                            { 0.65f, 0.45f },
                            { 0.95f, 0.35f },
                            { 0.05f, 0.65f },
                            { 0.35f, 0.55f },
                            { 0.65f, 0.55f },
                            { 0.95f, 0.65f },
                            { 0.05f, 0.95f },
                            { 0.35f, 0.85f },
                            { 0.65f, 0.85f },
                            { 0.95f, 0.95f }};

    posVector.resize(8 * 3);
    std::memcpy(&posVector[0], dfltPositions, 8 * 3 * sizeof(float));

    uvVector.resize(16 * 2);
    std::memcpy(&uvVector[0], dfltUVs, 16 * 2 * sizeof(float));

    //
    //  Initialize a Far::TopologyDescriptor, from which to create
    //  the Far::TopologyRefiner:
    //
    typedef Far::TopologyDescriptor Descriptor;

    Descriptor::FVarChannel uvChannel;
    uvChannel.numValues    = dfltNumUVs;
    uvChannel.valueIndices = dfltFaceFVars;

    Descriptor topDescriptor;
    topDescriptor.numVertices        = dfltNumVerts;
    topDescriptor.numFaces           = dfltNumFaces;
    topDescriptor.numVertsPerFace    = dfltFaceSizes;
    topDescriptor.vertIndicesPerFace = dfltFaceVerts;
    topDescriptor.numFVarChannels    = 1;
    topDescriptor.fvarChannels       = &uvChannel;

    Sdc::SchemeType schemeType = Sdc::SCHEME_CATMARK;

    Sdc::Options schemeOptions;
    schemeOptions.SetVtxBoundaryInterpolation(
                            Sdc::Options::VTX_BOUNDARY_EDGE_ONLY);
    schemeOptions.SetFVarLinearInterpolation(
                            Sdc::Options::FVAR_LINEAR_CORNERS_ONLY);

    typedef Far::TopologyRefinerFactory<Descriptor> RefinerFactory;

    Far::TopologyRefiner * topRefiner =
        RefinerFactory::Create(topDescriptor,
            RefinerFactory::Options(schemeType, schemeOptions));
    assert(topRefiner);
    return topRefiner;
}

//
//  Create a TopologyRefiner from a specified Obj file:
//
Far::TopologyRefiner *
readTopologyRefiner(std::string const & objFileName,
                    Sdc::SchemeType schemeType,
                    std::vector<float> & posVector,
                    std::vector<float> & uvVector) {

    const char *  filename = objFileName.c_str();
    const Shape * shape = 0;

    std::ifstream ifs(filename);
    if (ifs) {
        std::stringstream ss;
        ss << ifs.rdbuf();
        ifs.close();
        std::string shapeString = ss.str();

        shape = Shape::parseObj(
            shapeString.c_str(), ConvertSdcTypeToShapeScheme(schemeType), false);
        if (shape == 0) {
            fprintf(stderr,
                "Error:  Cannot create Shape from Obj file '%s'\n", filename);
            return 0;
        }
    } else {
        fprintf(stderr, "Error:  Cannot open Obj file '%s'\n", filename);
        return 0;
    }

    Sdc::SchemeType sdcType    = GetSdcType(*shape);
    Sdc::Options    sdcOptions = GetSdcOptions(*shape);

    Far::TopologyRefiner * refiner = Far::TopologyRefinerFactory<Shape>::Create(
        *shape, Far::TopologyRefinerFactory<Shape>::Options(sdcType, sdcOptions));
    if (refiner == 0) {
        fprintf(stderr,
            "Error:  Unable to construct TopologyRefiner from Obj file '%s'\n",
            filename);
        return 0;
    }

    int numVertices = refiner->GetNumVerticesTotal();
    posVector.resize(numVertices * 3);
    std::memcpy(&posVector[0], &shape->verts[0], 3*numVertices*sizeof(float));

    uvVector.resize(0);
    if (refiner->GetNumFVarChannels()) {
        int numUVs = refiner->GetNumFVarValuesTotal(0);
        uvVector.resize(numUVs * 2);
        std::memcpy(&uvVector[0], &shape->uvs[0], 2 * numUVs*sizeof(float));
    }

    delete shape;
    return refiner;
}

Far::TopologyRefiner *
createTopologyRefiner(std::string const & objFileName,
                      Sdc::SchemeType schemeType,
                      std::vector<float> & posVector,
                      std::vector<float> & uvVector) {

    if (objFileName.empty()) {
        return dfltTopologyRefiner(posVector, uvVector);
    } else {
        return readTopologyRefiner(objFileName, schemeType,
                                   posVector, uvVector);
    }
}

} // end namespace
