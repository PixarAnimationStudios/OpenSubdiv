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
// refine it uniformly and then interpolate additional sets of primvar data.
//

#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/primvarRefiner.h>

#include <cstdio>

//------------------------------------------------------------------------------
// Vertex container implementation.
//
// We are adding a per-vertex color attribute to our primvar data.  While they
// are separate properties and exist in separate buffers (as when read from an
// Alembic file) they are both of the form float[3] and so we can use the same
// underlying type.
//
// While color and position may be the same, we'll make the color a "varying"
// primvar, e.g. it is constrained to being linearly interpolated between
// vertices, rather than smoothly like position and other vertex data.
//
struct Point3 {

    // Minimal required interface ----------------------
    Point3() { }

    void Clear( void * =0 ) {
        _point[0]=_point[1]=_point[2]=0.0f;
    }

    void AddWithWeight(Point3 const & src, float weight) {
        _point[0]+=weight*src._point[0];
        _point[1]+=weight*src._point[1];
        _point[2]+=weight*src._point[2];
    }

    // Public interface ------------------------------------
    void SetPoint(float x, float y, float z) {
        _point[0]=x;
        _point[1]=y;
        _point[2]=z;
    }

    const float * GetPoint() const {
        return _point;
    }

private:
    float _point[3];
};

typedef Point3 VertexPosition;
typedef Point3 VertexColor;

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

    // Uniformly refine the topology up to 'maxlevel'
    refiner->RefineUniform(Far::TopologyRefiner::UniformOptions(maxlevel));

    // Allocate buffers for vertex primvar data.
    //
    // We assume we received the coarse data for the mesh in separate buffers
    // from some other source, e.g. an Alembic file.  Meanwhile, we want buffers
    // for the last/finest subdivision level to persist.  We have no interest
    // in the intermediate levels.
    //
    // Determine the sizes for our needs:
    int nCoarseVerts = g_nverts;
    int nFineVerts   = refiner->GetLevel(maxlevel).GetNumVertices();
    int nTotalVerts  = refiner->GetNumVerticesTotal();
    int nTempVerts   = nTotalVerts - nCoarseVerts - nFineVerts;

    // Allocate and initialize the primvar data for the original coarse vertices:
    std::vector<VertexPosition> coarsePosBuffer(nCoarseVerts);
    std::vector<VertexColor>    coarseClrBuffer(nCoarseVerts);

    for (int i = 0; i < nCoarseVerts; ++i) {
        coarsePosBuffer[i].SetPoint(g_verts[i][0], g_verts[i][1], g_verts[i][2]);
        coarseClrBuffer[i].SetPoint(g_colors[i][0], g_colors[i][1], g_colors[i][2]);
    }

    // Allocate intermediate and final storage to be populated:
    std::vector<VertexPosition> tempPosBuffer(nTempVerts);
    std::vector<VertexPosition> finePosBuffer(nFineVerts);

    std::vector<VertexColor> tempClrBuffer(nTempVerts);
    std::vector<VertexColor> fineClrBuffer(nFineVerts);

    // Interpolate all primvar data -- separate buffers can be populated on
    // separate threads if desired:
    VertexPosition * srcPos = &coarsePosBuffer[0];
    VertexPosition * dstPos = &tempPosBuffer[0];

    VertexColor * srcClr = &coarseClrBuffer[0];
    VertexColor * dstClr = &tempClrBuffer[0];

    Far::PrimvarRefiner primvarRefiner(*refiner);

    for (int level = 1; level < maxlevel; ++level) {
        primvarRefiner.Interpolate(       level, srcPos, dstPos);
        primvarRefiner.InterpolateVarying(level, srcClr, dstClr);

        srcPos = dstPos, dstPos += refiner->GetLevel(level).GetNumVertices();
        srcClr = dstClr, dstClr += refiner->GetLevel(level).GetNumVertices();
    }

    // Interpolate the last level into the separate buffers for our final data:
    primvarRefiner.Interpolate(       maxlevel, srcPos, finePosBuffer);
    primvarRefiner.InterpolateVarying(maxlevel, srcClr, fineClrBuffer);


    { // Visualization with Maya : print a MEL script that generates colored
      // particles at the location of the refined vertices (don't forget to
      // turn shading on in the viewport to see the colors)

        int nverts = nFineVerts;

        // Output particle positions
        printf("particle ");
        for (int vert = 0; vert < nverts; ++vert) {
            float const * pos = finePosBuffer[vert].GetPoint();
            printf("-p %f %f %f\n", pos[0], pos[1], pos[2]);
        }
        printf(";\n");

        // Set particle point size (20 -- very large)
        printf("addAttr -is true -ln \"pointSize\" -at long -dv 20 particleShape1;\n");

        // Add per-particle color attribute ('rgbPP')
        printf("addAttr -ln \"rgbPP\" -dt vectorArray particleShape1;\n");

        // Set per-particle color values from our primvar data
        printf("setAttr \"particleShape1.rgbPP\" -type \"vectorArray\" %d ", nverts);
        for (int vert = 0; vert < nverts; ++vert) {
            float const * color = fineClrBuffer[vert].GetPoint();
            printf("%f %f %f\n", color[0], color[1], color[2]);
        }
        printf(";\n");
    }

    delete refiner;
    return EXIT_SUCCESS;
}

//------------------------------------------------------------------------------
// Creates Far::TopologyRefiner from raw geometry
//
// see tutorial_1_1 for more details
//
static Far::TopologyRefiner *
createFarTopologyRefiner() {

    // Populate a topology descriptor with our raw data

    typedef Far::TopologyDescriptor Descriptor;

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
