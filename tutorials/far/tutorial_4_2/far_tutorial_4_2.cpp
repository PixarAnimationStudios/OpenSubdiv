//
//   Copyright 2013 Pixar
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


//------------------------------------------------------------------------------
// Tutorial description:
//
// This tutorial shows how to create and manipulate both 'vertex' and 'varying'
// Far::StencilTable to interpolate 2 primvar data buffers: vertex positions and
// vertex colors.
//

#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/stencilTable.h>
#include <opensubdiv/far/stencilTableFactory.h>

#include <cstdio>
#include <cstring>

//------------------------------------------------------------------------------
// Vertex container implementation.
//
struct Vertex {

    // Minimal required interface ----------------------
    Vertex() { }

    Vertex(Vertex const & src) {
        _data[0] = src._data[0];
        _data[1] = src._data[1];
        _data[2] = src._data[2];
    }

    void Clear( void * =0 ) {
        _data[0]=_data[1]=_data[2]=0.0f;
    }

    void AddWithWeight(Vertex const & src, float weight) {
        _data[0]+=weight*src._data[0];
        _data[1]+=weight*src._data[1];
        _data[2]+=weight*src._data[2];
    }

    // Public interface ------------------------------------
    float const * GetData() const {
        return _data;
    }

private:
    float _data[3];
};

//------------------------------------------------------------------------------
// Cube geometry from catmark_cube.h

static float g_verts[24] = {-0.5f, -0.5f,  0.5f,
                             0.5f, -0.5f,  0.5f,
                            -0.5f,  0.5f,  0.5f,
                             0.5f,  0.5f,  0.5f,
                            -0.5f,  0.5f, -0.5f,
                             0.5f,  0.5f, -0.5f,
                            -0.5f, -0.5f, -0.5f,
                             0.5f, -0.5f, -0.5f };

// Per-vertex RGB color data
static float g_colors[24] = { 1.0f, 0.0f, 0.5f,
                              0.0f, 1.0f, 0.0f,
                              0.0f, 0.0f, 1.0f,
                              1.0f, 1.0f, 1.0f,
                              1.0f, 1.0f, 0.0f,
                              0.0f, 1.0f, 1.0f,
                              1.0f, 0.0f, 1.0f,
                              0.0f, 0.0f, 0.0f };


static int g_nverts = 8,
           g_nfaces = 6;

static int g_vertsperface[6] = { 4, 4, 4, 4, 4, 4 };

static int g_vertIndices[24] = { 0, 1, 3, 2,
                                 2, 3, 5, 4,
                                 4, 5, 7, 6,
                                 6, 7, 1, 0,
                                 1, 7, 5, 3,
                                 6, 0, 2, 4  };

using namespace OpenSubdiv;

static Far::TopologyRefiner * createTopologyRefiner();

//------------------------------------------------------------------------------
int main(int, char **) {

    // Generate a Far::TopologyRefiner (see tutorial_1_1 for details).
    Far::TopologyRefiner * refiner = createTopologyRefiner();


    // Uniformly refine the topology up to 'maxlevel'.
    int maxlevel = 4;
    refiner->RefineUniform(Far::TopologyRefiner::UniformOptions(maxlevel));

    int nverts = refiner->GetLevel(maxlevel).GetNumVertices();

    // Use the Far::StencilTable factory to create discrete stencil table
    Far::StencilTableFactory::Options options;
    options.generateIntermediateLevels=false; // only the highest refinement level.
    options.generateOffsets=true;

    //
    // Vertex primvar data
    //

        // Create stencils table for 'vertex' interpolation
        options.interpolationMode=Far::StencilTableFactory::INTERPOLATE_VERTEX;

        Far::StencilTable const * vertexStencils =
            Far::StencilTableFactory::Create(*refiner, options);
        assert(nverts==vertexStencils->GetNumStencils());

        // Allocate vertex primvar buffer (1 stencil for each vertex)
        std::vector<Vertex> vertexBuffer(vertexStencils->GetNumStencils());

        // Use the cube vertex positions as 'vertex' primvar data
        Vertex * vertexCVs = reinterpret_cast<Vertex *>(g_verts);

    //
    // Varying primvar data
    //

        // Create stencils table for 'varying' interpolation
        options.interpolationMode=Far::StencilTableFactory::INTERPOLATE_VARYING;

        Far::StencilTable const * varyingStencils =
            Far::StencilTableFactory::Create(*refiner, options);
        assert(nverts==varyingStencils->GetNumStencils());

        // Allocate varying primvar buffer (1 stencil for each vertex)
        std::vector<Vertex> varyingBuffer(varyingStencils->GetNumStencils());

        // Use per-vertex array of RGB colors as 'varying' primvar data
        Vertex * varyingCVs = reinterpret_cast<Vertex *>(g_colors);

    delete refiner;

    //
    // Apply stencils (in frame loop)
    //

    { // This section would be applied every frame after control vertices have
      // been moved.

        // Apply stencils on the control vertex data to update the primvar data
        // of the refined vertices.

        vertexStencils->UpdateValues(vertexCVs, &vertexBuffer[0]);

        varyingStencils->UpdateValues(varyingCVs, &varyingBuffer[0]);
    }

    { // Visualization with Maya : print a MEL script that generates particles
      // at the location of the refined vertices

        printf("particle ");
        for (int vert=0; vert<(int)nverts; ++vert) {
            float const * pos = vertexBuffer[vert].GetData();
            printf("-p %f %f %f\n", pos[0], pos[1], pos[2]);
        }
        printf("-c 1;\n");

        // Set particle point size (20 -- very large)
        printf("addAttr -is true -ln \"pointSize\" -at long -dv 20 particleShape1;\n");

        // Add per-particle color attribute ('rgbPP')
        printf("addAttr -ln \"rgbPP\" -dt vectorArray particleShape1;\n");

        // Set per-particle color values from our 'varying' primvar data
        printf("setAttr \"particleShape1.rgbPP\" -type \"vectorArray\" %d ", nverts);
        for (int vert=0; vert<nverts; ++vert) {
            float const * color = varyingBuffer[vert].GetData();
            printf("%f %f %f\n", color[0], color[1], color[2]);
        }
        printf(";\n");
    }

    delete vertexStencils;
    delete varyingStencils;
    return EXIT_SUCCESS;
}

//------------------------------------------------------------------------------
static Far::TopologyRefiner *
createTopologyRefiner() {

    // Populate a topology descriptor with our raw data.

    typedef Far::TopologyDescriptor Descriptor;

    Sdc::SchemeType type = OpenSubdiv::Sdc::SCHEME_CATMARK;

    Sdc::Options options;
    options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_ONLY);

    Descriptor desc;
    desc.numVertices = g_nverts;
    desc.numFaces = g_nfaces;
    desc.numVertsPerFace = g_vertsperface;
    desc.vertIndicesPerFace = g_vertIndices;

    // Instantiate a Far::TopologyRefiner from the descriptor.
    Far::TopologyRefiner * refiner =
        Far::TopologyRefinerFactory<Descriptor>::Create(desc,
            Far::TopologyRefinerFactory<Descriptor>::Options(type, options));

    return refiner;
}

//------------------------------------------------------------------------------
