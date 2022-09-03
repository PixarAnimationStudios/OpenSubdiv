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
// This tutorial shows how to create and manipulate table of cascading stencils.
//
// We initialize a Far::TopologyRefiner with a cube and apply uniform
// refinement. We then use a Far::StencilTableFactory to generate a stencil
// table. We set the factory Options to not factorize intermediate levels,
// thus giving a table of "cascading" stencils.
//
// We then apply the stencils to the vertex position primvar data, and insert
// a hierarchical edit at level 1. This edit is smoothed by the application
// of the subsequent stencil cascades.
//
// The results are dumped into an OBJ file that shows the intermediate levels
// of refinement of the original cube.
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

    float * GetPosition() {
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
    int maxlevel = 4;
    refiner->RefineUniform(Far::TopologyRefiner::UniformOptions(maxlevel));

    // Use the Far::StencilTable factory to create cascading stencil table
    // note: we want stencils for each refinement level
    //       "cascade" mode is achieved by setting "factorizeIntermediateLevels"
    //       to false
    Far::StencilTableFactory::Options options;
    options.generateIntermediateLevels=true;
    options.factorizeIntermediateLevels=false;
    options.generateOffsets=true;

    Far::StencilTable const * stencilTable =
        Far::StencilTableFactory::Create(*refiner, options);

    std::vector<Vertex> vertexBuffer(refiner->GetNumVerticesTotal()-g_nverts);

    Vertex * destVerts = &vertexBuffer[0];

    int start = 0, end = 0; // stencil batches for each level of subdivision
    for (int level=0; level<maxlevel; ++level) {

        int nverts = refiner->GetLevel(level+1).GetNumVertices();

        Vertex const * srcVerts = reinterpret_cast<Vertex *>(g_verts);
        if (level>0) {
             srcVerts = &vertexBuffer[start];
        }

        start = end;
        end += nverts;

        stencilTable->UpdateValues(srcVerts, destVerts, start, end);
        
        // apply 2 hierarchical edits on level 1 vertices
        if (level==1) {
            float * pos = destVerts[start+5].GetPosition();
            pos[1] += 0.5f;            

            pos = destVerts[start+20].GetPosition();
            pos[0] += 0.25f;
        }
    }


    { // Output OBJ of the highest level refined -----------

        Vertex * verts = &vertexBuffer[0];

        // Print vertex positions
        for (int level=1, firstvert=0; level<=maxlevel; ++level) {

            Far::TopologyLevel const & refLevel = refiner->GetLevel(level);

            printf("g level_%d\n", level);

            int nverts = refLevel.GetNumVertices();
            for (int vert=0; vert<nverts; ++vert) {
                float const * pos = verts[vert].GetPosition();
                printf("v %f %f %f\n", pos[0], pos[1], pos[2]);
            }
            verts += nverts;
 
            // Print faces
            for (int face=0; face<refLevel.GetNumFaces(); ++face) {

                Far::ConstIndexArray fverts = refLevel.GetFaceVertices(face);

                // all refined Catmark faces should be quads
                assert(fverts.size()==4);

                printf("f ");
                for (int vert=0; vert<fverts.size(); ++vert) {
                    printf("%d ", fverts[vert]+firstvert+1); // OBJ uses 1-based arrays...
                }
                printf("\n");
            }
            firstvert+=nverts;
        }
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
