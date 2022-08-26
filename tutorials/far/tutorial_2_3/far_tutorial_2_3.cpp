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
// NOTE: The following approaches are approximations to compute smooth normals,
//       for highest fidelity patches should be used for positions and normals, 
//       which form the true limit surface.
//
// Building on tutorial 3, this example shows how to instantiate a simple mesh,
// refine it uniformly, interpolate both 'vertex' and 'face-varying'
// primvar data, and finally calculate approximated smooth normals. 
// The resulting interpolated data is output in 'obj' format.
//
// Currently, this tutorial supports 3 methods to approximate smooth normals:
// 
//     CrossTriangle : Calculates smooth normals (accumulating per vertex) using
//                     3 verts to generate 2 vectors. This approximation has
//                     trouble when working with quads (which can be non-planar)
//                     since it only takes into account half of each face. 
//
//     CrossQuad     : Calculates smooth normals (accumulating per vertex) 
//                     but this time, instead of taking into account only 3 verts
//                     it creates 2 vectors crossing the quad.
//                     This approximation builds upon CrossTriangle but takes
//                     into account the 4 verts of the face.
//
//     Limit         : Calculates the normals at the limit for each vert
//                     at the last level of subdivision.
//                     These are the true limit normals, however, in this example
//                     they are used with verts that are not at the limit. 
//                     This can lead to new visual artifacts since the normals
//                     and the positions don't match. Additionally, this approach
//                     requires extra computation to calculate the limit normals.
//                     For this reason, we strongly suggest using  
//                     limit positions with limit normals.
//

#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/primvarRefiner.h>

#include <cstdio>

//------------------------------------------------------------------------------
// Math helpers.
//
//

// Returns the normalized version of the input vector
inline void
normalize(float *n) {
    float rn = 1.0f/sqrtf(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
    n[0] *= rn;
    n[1] *= rn;
    n[2] *= rn;
}

// Returns the cross product of \p v1 and \p v2.                                
void cross(float const *v1, float const *v2, float* vOut)
{                                                                                
    vOut[0] = v1[1] * v2[2] - v1[2] * v2[1];
    vOut[1] = v1[2] * v2[0] - v1[0] * v2[2];
    vOut[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

//------------------------------------------------------------------------------
// Face-varying implementation.
//
//
struct Vertex {

    // Minimal required interface ----------------------
    Vertex() { 
        Clear();
    }

    Vertex(Vertex const & src) {
        position[0] = src.position[0];
        position[1] = src.position[1];
        position[2] = src.position[2];
    }

    void Clear() {
        position[0]=position[1]=position[2]=0.0f;
    }

    void AddWithWeight(Vertex const & src, float weight) {
        position[0]+=weight*src.position[0];
        position[1]+=weight*src.position[1];
        position[2]+=weight*src.position[2];
    }

    // Public interface ------------------------------------
    void SetPosition(float x, float y, float z) {
        position[0]=x;
        position[1]=y;
        position[2]=z;
    }

    const float * GetPosition() const {
        return position;
    }

    float position[3];
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

// Approximation methods for smooth normal computations
enum NormalApproximation
{
    CrossTriangle,
    CrossQuad,
    Limit
};

//------------------------------------------------------------------------------
int main(int argc, char ** argv) {

    const int maxlevel = 2;
    enum NormalApproximation normalApproximation = CrossTriangle;

    // Parsing command line parameters to see if the user wants to use a  
    // specific method to calculate normals
    for (int i = 1; i < argc; ++i) {

        if (strstr(argv[i], "-limit")) {
            normalApproximation = Limit;
        } else if (!strcmp(argv[i], "-crossquad")) {
            normalApproximation = CrossQuad;
        } else if (!strcmp(argv[i], "-crosstriangle")) {
            normalApproximation = CrossTriangle;
        } else {
            printf("Parameters : \n");
            printf("  -crosstriangle : use the cross product of vectors\n");
            printf("                   generated from 3 verts (default).\n");
            printf("  -crossquad     : use the cross product of vectors\n");
            printf("                   generated from 4 verts.\n");
            printf("  -limit         : use normals calculated from the limit.\n");
            return 0;
        }
    }

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
   
    // Create a face-varying channel descriptor
    const int numChannels  = 2;
    const int channelUV    = 0;
    const int channelColor = 1;
    Descriptor::FVarChannel channels[numChannels];
    channels[channelUV].numValues = g_nuvs;
    channels[channelUV].valueIndices = g_uvIndices;
    channels[channelColor].numValues = g_ncolors;
    channels[channelColor].valueIndices = g_colorIndices;

    // Add the channel topology to the main descriptor
    desc.numFVarChannels = numChannels;
    desc.fvarChannels = channels;

    // Instantiate a Far::TopologyRefiner from the descriptor
    Far::TopologyRefiner * refiner =
        Far::TopologyRefinerFactory<Descriptor>::Create(desc,
            Far::TopologyRefinerFactory<Descriptor>::Options(type, options));

    // Uniformly refine the topolgy up to 'maxlevel'
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

    // Allocate & initialize the first channel of 'face-varying' primvars (UVs)
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
    Vertex *          srcVert = verts;
    FVarVertexUV *    srcFVarUV = fvVertsUV;
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

    // Approximate normals
    Far::TopologyLevel const & refLastLevel = refiner->GetLevel(maxlevel);
    int nverts = refLastLevel.GetNumVertices();
    int nfaces = refLastLevel.GetNumFaces();
    int firstOfLastVerts = refiner->GetNumVerticesTotal() - nverts;

    std::vector<Vertex> normals(nverts);

    // Different ways to approximate smooth normals
    //
    // For details check the description at the beginning of the file
    if (normalApproximation == Limit) {

        // Approximation using the normal at the limit with verts that are 
        // not at the limit
        //
        // For details check the description at the beginning of the file

        std::vector<Vertex> fineLimitPos(nverts);
        std::vector<Vertex> fineDu(nverts);
        std::vector<Vertex> fineDv(nverts);

        primvarRefiner.Limit(&verts[firstOfLastVerts], fineLimitPos, fineDu, fineDv);
        
        for (int vert = 0; vert < nverts; ++vert) {
            float const * du = fineDu[vert].GetPosition();
            float const * dv = fineDv[vert].GetPosition();
            
            float norm[3];
            cross(du, dv, norm);
            normals[vert].SetPosition(norm[0], norm[1], norm[2]);
        }

    } else if (normalApproximation == CrossQuad) {

        // Approximate smooth normals by accumulating normal vectors computed as
        // the cross product of two vectors generated by the 4 verts that 
        // form each quad
        //
        // For details check the description at the beginning of the file

        for (int f = 0; f < nfaces; f++) {
            Far::ConstIndexArray faceVertices = refLastLevel.GetFaceVertices(f);

            // We will use the first three verts to calculate a normal
            const float * v0 = verts[ firstOfLastVerts + faceVertices[0] ].GetPosition();
            const float * v1 = verts[ firstOfLastVerts + faceVertices[1] ].GetPosition();
            const float * v2 = verts[ firstOfLastVerts + faceVertices[2] ].GetPosition();
            const float * v3 = verts[ firstOfLastVerts + faceVertices[3] ].GetPosition();

            // Calculate the cross product between the vectors formed by v1-v0 and
            // v2-v0, and then normalize the result
            float normalCalculated [] = {0.0,0.0,0.0};
            float a[3] = { v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2] };
            float b[3] = { v3[0]-v1[0], v3[1]-v1[1], v3[2]-v1[2] };          
            cross(a, b, normalCalculated);
            normalize(normalCalculated);

            // Accumulate that normal on all verts that are part of that face
            for(int vInFace = 0; vInFace < faceVertices.size() ; vInFace++ ) {

                int vertexIndex = faceVertices[vInFace];
                normals[vertexIndex].position[0] += normalCalculated[0];
                normals[vertexIndex].position[1] += normalCalculated[1];
                normals[vertexIndex].position[2] += normalCalculated[2];
            }
        }

    } else if (normalApproximation == CrossTriangle) {

        // Approximate smooth normals by accumulating normal vectors computed as
        // the cross product of two vectors generated by 3 verts of the quad
        //
        // For details check the description at the beginning of the file

        for (int f = 0; f < nfaces; f++) {
            Far::ConstIndexArray faceVertices = refLastLevel.GetFaceVertices(f);

            // We will use the first three verts to calculate a normal
            const float * v0 = verts[ firstOfLastVerts + faceVertices[0] ].GetPosition();
            const float * v1 = verts[ firstOfLastVerts + faceVertices[1] ].GetPosition();
            const float * v2 = verts[ firstOfLastVerts + faceVertices[2] ].GetPosition();

            // Calculate the cross product between the vectors formed by v1-v0 and
            // v2-v0, and then normalize the result
            float normalCalculated [] = {0.0,0.0,0.0};
            float a[3] = { v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2] };
            float b[3] = { v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2] };
            cross(a, b, normalCalculated);
            normalize(normalCalculated);

            // Accumulate that normal on all verts that are part of that face
            for(int vInFace = 0; vInFace < faceVertices.size() ; vInFace++ ) {

                int vertexIndex = faceVertices[vInFace];
                normals[vertexIndex].position[0] += normalCalculated[0];
                normals[vertexIndex].position[1] += normalCalculated[1];
                normals[vertexIndex].position[2] += normalCalculated[2];
            }
        }
    }

    // Finally we just need to normalize the accumulated normals
    for (int vert = 0; vert < nverts; ++vert) {
        normalize(&normals[vert].position[0]);
    }
   
    { // Output OBJ of the highest level refined -----------

        // Print vertex positions
        for (int vert = 0; vert < nverts; ++vert) {
            float const * pos = verts[firstOfLastVerts + vert].GetPosition();
            printf("v %f %f %f\n", pos[0], pos[1], pos[2]);
        }
        
        // Print vertex normals
        for (int vert = 0; vert < nverts; ++vert) {
            float const * pos = normals[vert].GetPosition();
            printf("vn %f %f %f\n", pos[0], pos[1], pos[2]);
        }

        // Print uvs
        int nuvs   = refLastLevel.GetNumFVarValues(channelUV);
        int firstOfLastUvs = refiner->GetNumFVarValuesTotal(channelUV) - nuvs;
        for (int fvvert = 0; fvvert < nuvs; ++fvvert) {
            FVarVertexUV const & uv = fvVertsUV[firstOfLastUvs + fvvert];
            printf("vt %f %f\n", uv.u, uv.v);
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
                printf("%d/%d/%d ", fverts[vert]+1, fuvs[vert]+1, fverts[vert]+1);
            }
            printf("\n");
        }
    }

    delete refiner;
    return EXIT_SUCCESS;
}
//------------------------------------------------------------------------------
