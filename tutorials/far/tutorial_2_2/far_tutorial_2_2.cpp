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

#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/primvarRefiner.h>

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

    const float * GetPosition() const {
        return _position;
    }

private:
    float _position[3];
};

//------------------------------------------------------------------------------
// Face-varying container implementation.
//
// We are using a uv texture layout as a 'face-varying' primitive variable
// attribute. Because face-varying data is specified 'per-face-per-vertex',
// we cannot use the same container that we use for 'vertex' or 'varying'
// data. We specify a new container, which only carries (u,v) coordinates.
// Similarly to our 'Vertex' container, we add a minimalistic interpolation
// interface with a 'Clear()' and 'AddWithWeight()' methods.
//
struct FVarVertexUV {

    // Minimal required interface ----------------------
    void Clear() {
        u=v=0.0f;
    }

    void AddWithWeight(FVarVertexUV const & src, float weight) {
        u += weight * src.u;
        v += weight * src.v;
    }

    // Basic 'uv' layout channel
    float u,v;
};

struct FVarVertexColor {

    // Minimal required interface ----------------------
    void Clear() {
        r=g=b=a=0.0f;
    }

    void AddWithWeight(FVarVertexColor const & src, float weight) {
        r += weight * src.r;
        g += weight * src.g;
        b += weight * src.b;
        a += weight * src.a;
    }

    // Basic 'color' layout channel
    float r,g,b,a;
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

// 'face-varying' primitive variable data & topology for UVs
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

// 'face-varying' primitive variable data & topology for color
static float g_colors[24][4] = {{1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 0.0, 0.0, 1.0},
                                {1.0, 0.0, 0.0, 1.0},
                                {1.0, 0.0, 0.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0}};

static int g_ncolors = 24;

static int g_colorIndices[24] = { 0,  3,  9,  6,
                                  7, 10, 15, 12, 
                                 13, 16, 21, 18,
                                 19, 22,  4,  1,
                                  5, 23, 17, 11,
                                 20,  2,  8, 14 };

using namespace OpenSubdiv;

//------------------------------------------------------------------------------
int main(int, char **) {

    int maxlevel = 3;

    typedef Far::TopologyDescriptor Descriptor;

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

    int channelUV = 0;
    int channelColor = 1;
    
    // Create a face-varying channel descriptor
    Descriptor::FVarChannel channels[2];
    channels[channelUV].numValues = g_nuvs;
    channels[channelUV].valueIndices = g_uvIndices;
    channels[channelColor].numValues = g_ncolors;
    channels[channelColor].valueIndices = g_colorIndices;

    // Add the channel topology to the main descriptor
    desc.numFVarChannels = 2;
    desc.fvarChannels = channels;

    // Instantiate a Far::TopologyRefiner from the descriptor
    Far::TopologyRefiner * refiner =
        Far::TopologyRefinerFactory<Descriptor>::Create(desc,
            Far::TopologyRefinerFactory<Descriptor>::Options(type, options));

    // Uniformly refine the topology up to 'maxlevel'
    // note: fullTopologyInLastLevel must be true to work with face-varying data
    {
        Far::TopologyRefiner::UniformOptions refineOptions(maxlevel);
        refineOptions.fullTopologyInLastLevel = true;
        refiner->RefineUniform(refineOptions);
    }

    // Allocate and initialize the 'vertex' primvar data (see tutorial 2 for
    // more details).
    std::vector<Vertex> vbuffer(refiner->GetNumVerticesTotal());
    Vertex * verts = &vbuffer[0];

    for (int i=0; i<g_nverts; ++i) {
        verts[i].SetPosition(g_verts[i][0], g_verts[i][1], g_verts[i][2]);
    }

    // Allocate and initialize the first channel of 'face-varying' primvar data (UVs)
    std::vector<FVarVertexUV> fvBufferUV(refiner->GetNumFVarValuesTotal(channelUV));
    FVarVertexUV * fvVertsUV = &fvBufferUV[0];
    for (int i=0; i<g_nuvs; ++i) {
        fvVertsUV[i].u = g_uvs[i][0];
        fvVertsUV[i].v = g_uvs[i][1];
    }

    // Allocate & interpolate the 'face-varying' primvar data (colors)
    std::vector<FVarVertexColor> fvBufferColor(refiner->GetNumFVarValuesTotal(channelColor));
    FVarVertexColor * fvVertsColor = &fvBufferColor[0];
    for (int i=0; i<g_ncolors; ++i) {
        fvVertsColor[i].r = g_colors[i][0];
        fvVertsColor[i].g = g_colors[i][1];
        fvVertsColor[i].b = g_colors[i][2];
        fvVertsColor[i].a = g_colors[i][3];
    }

    // Interpolate both vertex and face-varying primvar data
    Far::PrimvarRefiner primvarRefiner(*refiner);

    Vertex *     srcVert = verts;
    FVarVertexUV * srcFVarUV = fvVertsUV;
    FVarVertexColor * srcFVarColor = fvVertsColor;

    for (int level = 1; level <= maxlevel; ++level) {
        Vertex *     dstVert = srcVert + refiner->GetLevel(level-1).GetNumVertices();
        FVarVertexUV * dstFVarUV = srcFVarUV + refiner->GetLevel(level-1).GetNumFVarValues(channelUV);
        FVarVertexColor * dstFVarColor = srcFVarColor + refiner->GetLevel(level-1).GetNumFVarValues(channelColor);

        primvarRefiner.Interpolate(level, srcVert, dstVert);
        primvarRefiner.InterpolateFaceVarying(level, srcFVarUV, dstFVarUV, channelUV);
        primvarRefiner.InterpolateFaceVarying(level, srcFVarColor, dstFVarColor, channelColor);

        srcVert = dstVert;
        srcFVarUV = dstFVarUV;
        srcFVarColor = dstFVarColor;
    }


    { // Output OBJ of the highest level refined -----------

        Far::TopologyLevel const & refLastLevel = refiner->GetLevel(maxlevel);

        int nverts = refLastLevel.GetNumVertices();
        int nuvs   = refLastLevel.GetNumFVarValues(channelUV);
        int ncolors= refLastLevel.GetNumFVarValues(channelColor);
        int nfaces = refLastLevel.GetNumFaces();

        // Print vertex positions
        int firstOfLastVerts = refiner->GetNumVerticesTotal() - nverts;

        for (int vert = 0; vert < nverts; ++vert) {
            float const * pos = verts[firstOfLastVerts + vert].GetPosition();
            printf("v %f %f %f\n", pos[0], pos[1], pos[2]);
        }

        // Print uvs
        int firstOfLastUvs = refiner->GetNumFVarValuesTotal(channelUV) - nuvs;

        for (int fvvert = 0; fvvert < nuvs; ++fvvert) {
            FVarVertexUV const & uv = fvVertsUV[firstOfLastUvs + fvvert];
            printf("vt %f %f\n", uv.u, uv.v);
        }

        // Print colors
        int firstOfLastColors = refiner->GetNumFVarValuesTotal(channelColor) - ncolors;

        for (int fvvert = 0; fvvert < ncolors; ++fvvert) {
            FVarVertexColor const & c = fvVertsColor[firstOfLastColors + fvvert];
            printf("c %f %f %f %f\n", c.r, c.g, c.b, c.a);
        }

        // Print faces
        for (int face = 0; face < nfaces; ++face) {

            Far::ConstIndexArray fverts = refLastLevel.GetFaceVertices(face);
            Far::ConstIndexArray fuvs   = refLastLevel.GetFaceFVarValues(face, channelUV);

            // all refined Catmark faces should be quads
            assert(fverts.size()==4 && fuvs.size()==4);

            printf("f ");
            for (int vert=0; vert<fverts.size(); ++vert) {
                // OBJ uses 1-based arrays...
                printf("%d/%d ", fverts[vert]+1, fuvs[vert]+1);
            }
            printf("\n");
        }
    }

    delete refiner;
    return EXIT_SUCCESS;
}
//------------------------------------------------------------------------------
