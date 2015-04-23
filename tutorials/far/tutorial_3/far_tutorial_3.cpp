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
// Building on tutorial 0, this example shows how to instantiate a simple mesh,
// refine it uniformly and then interpolate both 'vertex' and 'face-varying'
// primvar data.
// The resulting interpolated data is output as an 'obj' file, with the
// 'face-varying' data recorded in the uv texture layout.
//

#include <opensubdiv/far/topologyRefinerFactory.h>

#include <cstdio>

//------------------------------------------------------------------------------
// Face-varying implementation.
//
//
struct Vertex {

    // Minimal required interface ----------------------
    Vertex() { }

    Vertex(Vertex const & src) {
        _position[0] = src._position[0];
        _position[1] = src._position[1];
        _position[1] = src._position[1];
    }

    void Clear( void * =0 ) {
        _position[0]=_position[1]=_position[2]=0.0f;
    }

    void AddWithWeight(Vertex const & src, float weight) {
        _position[0]+=weight*src._position[0];
        _position[1]+=weight*src._position[1];
        _position[2]+=weight*src._position[2];
    }

    void AddVaryingWithWeight(Vertex const &, float) { }

    // Public interface ------------------------------------
    void SetPosition(float x, float y, float z) {
        _position[0]=x;
        _position[1]=y;
        _position[2]=z;
    }

    const float * GetPosition() const {
        return _position;
    }

private:
    float _position[3];
};

//------------------------------------------------------------------------------
// Face-varying container implementation.
//
// We are using a uv texture layout as a 'face-varying' primtiive variable
// attribute. Because face-varying data is specified 'per-face-per-vertex',
// we cannot use the same container that we use for 'vertex' or 'varying'
// data. We specify a new container, which only carries (u,v) coordinates.
// Similarly to our 'Vertex' container, we add a minimaliztic interpolation
// interface with a 'Clear()' and 'AddWithWeight()' methods.
//
struct FVarVertex {

    // Minimal required interface ----------------------
    void Clear() {
        u=v=0.0f;
    }

    void AddWithWeight(FVarVertex const & src, float weight) {
        u += weight * src.u;
        v += weight * src.v;
    }

    // Basic 'uv' layout channel
    float u,v;
};

//------------------------------------------------------------------------------
// Cube geometry from catmark_cube.h

// 'vertex' primitive variable data & topology
static float g_verts[8][3] = {{ -0.5f, -0.5f,  0.5f },
                              {  0.5f, -0.5f,  0.5f },
                              { -0.5f,  0.5f,  0.5f },
                              {  0.5f,  0.5f,  0.5f },
                              { -0.5f,  0.5f, -0.5f },
                              {  0.5f,  0.5f, -0.5f },
                              { -0.5f, -0.5f, -0.5f },
                              {  0.5f, -0.5f, -0.5f }};
static int g_nverts = 8,
           g_nfaces = 6;

static int g_vertsperface[6] = { 4, 4, 4, 4, 4, 4 };

static int g_vertIndices[24] = { 0, 1, 3, 2,
                                 2, 3, 5, 4,
                                 4, 5, 7, 6,
                                 6, 7, 1, 0,
                                 1, 7, 5, 3,
                                 6, 0, 2, 4  };


// 'face-varying' primitive variable data & topology
static float g_uvs[14][2] = {{ 0.375, 0.00 },
                             { 0.625, 0.00 },
                             { 0.375, 0.25 },
                             { 0.625, 0.25 },
                             { 0.375, 0.50 },
                             { 0.625, 0.50 },
                             { 0.375, 0.75 },
                             { 0.625, 0.75 },
                             { 0.375, 1.00 },
                             { 0.625, 1.00 },
                             { 0.875, 0.00 },
                             { 0.875, 0.25 },
                             { 0.125, 0.00 },
                             { 0.125, 0.25 }};

static int g_nuvs = 14;

static int g_uvIndices[24] = {  0,  1,  3,  2,
                                2,  3,  5,  4,
                                4,  5,  7,  6,
                                6,  7,  9,  8,
                                1, 10, 11,  3,
                               12,  0,  2, 13  };

using namespace OpenSubdiv;

//------------------------------------------------------------------------------
int main(int, char **) {

    int maxlevel = 3;

    typedef Far::TopologyRefinerFactoryBase::TopologyDescriptor Descriptor;

    Sdc::SchemeType type = OpenSubdiv::Sdc::SCHEME_CATMARK;

    Sdc::Options options;
    options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_ONLY);
    options.SetFVarLinearInterpolation(Sdc::Options::FVAR_LINEAR_NONE);

    // Populate a topology descriptor with our raw data
    Descriptor desc;
    desc.numVertices  = g_nverts;
    desc.numFaces     = g_nfaces;
    desc.numVertsPerFace = g_vertsperface;
    desc.vertIndicesPerFace  = g_vertIndices;

    // Create a face-varying channel descriptor
    Descriptor::FVarChannel uvs;
    uvs.numValues = g_nuvs;
    uvs.valueIndices = g_uvIndices;

    // Add the channel topology to the main descriptor
    desc.numFVarChannels = 1;
    desc.fvarChannels = & uvs;

    // Instantiate a FarTopologyRefiner from the descriptor
    Far::TopologyRefiner * refiner =
        Far::TopologyRefinerFactory<Descriptor>::Create(desc,
            Far::TopologyRefinerFactory<Descriptor>::Options(type, options));

    // Uniformly refine the topolgy up to 'maxlevel'
    // note: fullTopologyInLastLevel must be true to work with face-varying data
    {
        Far::TopologyRefiner::UniformOptions options(maxlevel);
        options.fullTopologyInLastLevel = true;
        refiner->RefineUniform(options);
    }

    // Allocate & interpolate the 'vertex' primvar data (see tutorial 2 for
    // more details).
    std::vector<Vertex> vbuffer(refiner->GetNumVerticesTotal());
    Vertex * verts = &vbuffer[0];

    int nCoarseVerts = g_nverts;
    for (int i=0; i<nCoarseVerts; ++i) {
        verts[i].SetPosition(g_verts[i][0], g_verts[i][1], g_verts[i][2]);
    }

    refiner->Interpolate(verts, verts + nCoarseVerts);


    // Allocate & interpolate the 'face-varying' primvar data
    int channel = 0,
        nCoarseFVVerts = refiner->GetNumFVarValues(0, channel);

    std::vector<FVarVertex> fvBuffer(refiner->GetNumFVarValuesTotal(channel));
    FVarVertex * fvVerts = &fvBuffer[0];

    for (int i=0; i<g_nuvs; ++i) {
        fvVerts[i].u = g_uvs[i][0];
        fvVerts[i].v = g_uvs[i][1];
    }

    refiner->InterpolateFaceVarying(fvVerts, fvVerts + nCoarseFVVerts, channel);


    { // Output OBJ of the highest level refined -----------

        // Print vertex positions
        for (int level=0, firstVert=0; level<=maxlevel; ++level) {

            if (level==maxlevel) {
                for (int vert=0; vert<refiner->GetNumVertices(level); ++vert) {
                    float const * pos = verts[firstVert+vert].GetPosition();
                    printf("v %f %f %f\n", pos[0], pos[1], pos[2]);
                }
            } else {
                firstVert += refiner->GetNumVertices(level);
            }
        }

        // Print uvs
        for (int level=0, firstVert=0; level<=maxlevel; ++level) {

            if (level==maxlevel) {
                for (int vert=0; vert<refiner->GetNumFVarValues(level, channel); ++vert) {
                    FVarVertex const & uv = fvVerts[firstVert+vert];
                    printf("vt %f %f\n", uv.u, uv.v);
                }
            } else {
                firstVert += refiner->GetNumFVarValues(level, channel);
            }
        }


        // Print faces
        for (int face=0; face<refiner->GetNumFaces(maxlevel); ++face) {

            Far::ConstIndexArray fverts = refiner->GetFaceVertices(maxlevel, face),
                                 fvverts = refiner->GetFVarFaceValues(maxlevel, face, channel);

            // all refined Catmark faces should be quads
            assert(fverts.size()==4 and fvverts.size()==4);

            printf("f ");
            for (int vert=0; vert<fverts.size(); ++vert) {
                // OBJ uses 1-based arrays...
                printf("%d/%d ", fverts[vert]+1, fvverts[vert]+1);
            }
            printf("\n");
        }
    }
}
//------------------------------------------------------------------------------
