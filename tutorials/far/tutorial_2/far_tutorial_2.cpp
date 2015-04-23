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
// refine it uniformly and then interpolate both 'vertex' and 'varying' primvar
// data.
//

#include <opensubdiv/far/topologyRefinerFactory.h>

#include <cstdio>

//------------------------------------------------------------------------------
// Vertex container implementation.
//
// We are adding a per-vertex color attribute to our Vertex interface. Unlike
// the position attribute however, the new color attribute is interpolated using
// the 'varying' mode of evaluation ('vertex' is bi-cubic, 'varying' is
// bi-linear). We also implemented the 'AddVaryingWithWeight()' method, which
// be performing the interpolation on the primvar data.
//
struct Vertex {

    // Minimal required interface ----------------------
    Vertex() { }

    void Clear( void * =0 ) {
        _position[0]=_position[1]=_position[2]=0.0f;
        _color[0]=_color[1]=_color[2]=0.0f;
    }

    void AddWithWeight(Vertex const & src, float weight) {
        _position[0]+=weight*src._position[0];
        _position[1]+=weight*src._position[1];
        _position[2]+=weight*src._position[2];
    }

    // The varying interpolation specialization must now be implemented.
    // Just like 'vertex' interpolation, it is a simple multiply-add.
    void AddVaryingWithWeight(Vertex const & src, float weight) {
        _color[0]+=weight*src._color[0];
        _color[1]+=weight*src._color[1];
        _color[2]+=weight*src._color[2];
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

    void SetColor(float x, float y, float z) {
        _color[0]=x;
        _color[1]=y;
        _color[2]=z;
    }

    const float * GetColor() const {
        return _color;
    }

private:
    float _position[3],
          _color[3];
};

//------------------------------------------------------------------------------
// Cube geometry from catmark_cube.h
static float g_verts[8][3] = {{ -0.5f, -0.5f,  0.5f },
                              {  0.5f, -0.5f,  0.5f },
                              { -0.5f,  0.5f,  0.5f },
                              {  0.5f,  0.5f,  0.5f },
                              { -0.5f,  0.5f, -0.5f },
                              {  0.5f,  0.5f, -0.5f },
                              { -0.5f, -0.5f, -0.5f },
                              {  0.5f, -0.5f, -0.5f }};

// Per-vertex RGB color data
static float g_colors[8][3] = {{ 1.0f, 0.0f, 0.5f },
                               { 0.0f, 1.0f, 0.0f },
                               { 0.0f, 0.0f, 1.0f },
                               { 1.0f, 1.0f, 1.0f },
                               { 1.0f, 1.0f, 0.0f },
                               { 0.0f, 1.0f, 1.0f },
                               { 1.0f, 0.0f, 1.0f },
                               { 0.0f, 0.0f, 0.0f }};

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

static Far::TopologyRefiner * createFarTopologyRefiner();

//------------------------------------------------------------------------------
int main(int, char **) {

    int maxlevel = 5;

    Far::TopologyRefiner * refiner = createFarTopologyRefiner();

    // Uniformly refine the topolgy up to 'maxlevel'
    refiner->RefineUniform(Far::TopologyRefiner::UniformOptions(maxlevel));

    // Allocate a buffer for vertex primvar data. The buffer length is set to
    // be the sum of all children vertices up to the highest level of refinement.
    std::vector<Vertex> vbuffer(refiner->GetNumVerticesTotal());
    Vertex * verts = &vbuffer[0];

    // Initialize coarse mesh primvar data
    int nCoarseVerts = g_nverts;
    for (int i=0; i<nCoarseVerts; ++i) {

        verts[i].SetPosition(g_verts[i][0], g_verts[i][1], g_verts[i][2]);

        verts[i].SetColor(g_colors[i][0], g_colors[i][1], g_colors[i][2]);
    }

    // Interpolate all primvar data - not that this will perform both 'vertex' and
    // 'varying' interpolation at once by calling each specialized method in our
    // Vertex class with the appropriate weights.
    refiner->Interpolate(verts, verts + nCoarseVerts);



    { // Visualization with Maya : print a MEL script that generates colored
      // particles at the location of the refined vertices (don't forget to
      // turn shading on in the viewport to see the colors)

        int nverts = refiner->GetNumVertices(maxlevel);

        // Position the 'verts' pointer to the first vertex of our 'maxlevel' level
        for (int level=0; level<maxlevel; ++level) {
            verts += refiner->GetNumVertices(level);
        }

        // Output particle positions
        printf("particle ");
        for (int vert=0; vert<nverts; ++vert) {
            float const * pos = verts[vert].GetPosition();
            printf("-p %f %f %f\n", pos[0], pos[1], pos[2]);
        }
        printf(";\n");

        // Set particle point size (20 -- very large)
        printf("addAttr -is true -ln \"pointSize\" -at long -dv 20 particleShape1;\n");

        // Add per-particle color attribute ('rgbPP')
        printf("addAttr -ln \"rgbPP\" -dt vectorArray particleShape1;\n");

        // Set per-particle color values from our 'varying' primvar data
        printf("setAttr \"particleShape1.rgbPP\" -type \"vectorArray\" %d ", nverts);
        for (int vert=0; vert<nverts; ++vert) {
            float const * color = verts[vert].GetColor();
            printf("%f %f %f\n", color[0], color[1], color[2]);
        }
        printf(";\n");
    }
}

//------------------------------------------------------------------------------
// Creates Far::TopologyRefiner from raw geometry
//
// see far_tutorial_0 for more details
//
static Far::TopologyRefiner *
createFarTopologyRefiner() {

    // Populate a topology descriptor with our raw data

    typedef Far::TopologyRefinerFactoryBase::TopologyDescriptor Descriptor;

    Sdc::SchemeType type = OpenSubdiv::Sdc::SCHEME_CATMARK;

    Sdc::Options options;
    options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_ONLY);

    Descriptor desc;
    desc.numVertices  = g_nverts;
    desc.numFaces     = g_nfaces;
    desc.numVertsPerFace = g_vertsperface;
    desc.vertIndicesPerFace  = g_vertIndices;

    // Instantiate a Far::TopologyRefiner from the descriptor
    Far::TopologyRefiner * refiner =
        Far::TopologyRefinerFactory<Descriptor>::Create(desc,
            Far::TopologyRefinerFactory<Descriptor>::Options(type, options));

    return refiner;
}
//------------------------------------------------------------------------------
