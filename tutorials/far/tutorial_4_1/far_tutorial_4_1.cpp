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
// This tutorial shows how to create and manipulate Far::StencilTable. We use
// the factorized stencils to interpolate vertex primvar data buffers.
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
        _position[0] = src._position[0];
        _position[1] = src._position[1];
        _position[2] = src._position[2];
    }

    void Clear( void * =0 ) {
        _position[0]=_position[1]=_position[2]=0.0f;
    }

    void AddWithWeight(Vertex const & src, float weight) {
        _position[0]+=weight*src._position[0];
        _position[1]+=weight*src._position[1];
        _position[2]+=weight*src._position[2];
    }

    // Public interface ------------------------------------
    void SetPosition(float x, float y, float z) {
        _position[0]=x;
        _position[1]=y;
        _position[2]=z;
    }

    float const * GetPosition() const {
        return _position;
    }

private:
    float _position[3];
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
    int maxlevel = 3;
    refiner->RefineUniform(Far::TopologyRefiner::UniformOptions(maxlevel));


    // Use the Far::StencilTable factory to create discrete stencil table
    // note: we only want stencils for the highest refinement level.
    Far::StencilTableFactory::Options options;
    options.generateIntermediateLevels=false;
    options.generateOffsets=true;
    
    Far::StencilTable const * stencilTable =
        Far::StencilTableFactory::Create(*refiner, options);

    // Allocate vertex primvar buffer (1 stencil for each vertex)
    int nstencils = stencilTable->GetNumStencils();
    std::vector<Vertex> vertexBuffer(nstencils);


    // Quick & dirty re-cast of the primvar data from our cube
    // (this is where you would drive shape deformations every frame)
    Vertex * controlValues = reinterpret_cast<Vertex *>(g_verts);

    { // This section would be applied every frame after control vertices have
      // been moved.

        // Apply stencils on the control vertex data to update the primvar data
        // of the refined vertices.
        stencilTable->UpdateValues(controlValues, &vertexBuffer[0]);
    }

    { // Visualization with Maya : print a MEL script that generates particles
      // at the location of the refined vertices

        printf("particle ");
        for (int i=0; i<(int)vertexBuffer.size(); ++i) {
            float const * pos = vertexBuffer[i].GetPosition();
            printf("-p %f %f %f\n", pos[0], pos[1], pos[2]);
        }
        printf("-c 1;\n");
    }

    delete refiner;
    delete stencilTable;
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
    return Far::TopologyRefinerFactory<Descriptor>::Create(desc,
                Far::TopologyRefinerFactory<Descriptor>::Options(type, options));

}

//------------------------------------------------------------------------------
