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

#include "glLoader.h"

#include <GLFW/glfw3.h>
GLFWwindow* g_window=0;
GLFWmonitor* g_primary=0;

#include <opensubdiv/osd/cpuGLVertexBuffer.h>

#include <opensubdiv/far/patchTableFactory.h>
#include <opensubdiv/far/stencilTable.h>
#include <opensubdiv/far/stencilTableFactory.h>
#include <opensubdiv/far/primvarRefiner.h>

#include "../../regression/common/far_utils.h"
#include "../common/stopwatch.h"
#include "../common/simple_math.h"
#include "../common/glUtils.h"
#include "../common/glControlMeshDisplay.h"
#include "../common/glHud.h"
#include "../common/glUtils.h"

#include "init_shapes.h"
#include "gl_mesh.h"
#include "gl_fontutils.h"

#include <typeinfo>
#include <cfloat>
#include <vector>
#include <set>
#include <fstream>
#include <sstream>
#include <cstdio>

#if _MSC_VER
    #define snprintf _snprintf
#endif


//------------------------------------------------------------------------------
int g_level = 3,
    g_currentShape = 7;

enum HudCheckBox { kHUD_CB_DISPLAY_CONTROL_MESH_EDGES,
                   kHUD_CB_DISPLAY_CONTROL_MESH_VERTS,
                   kHUD_CB_ANIMATE_VERTICES,
                   kHUD_CB_DISPLAY_PATCH_COLOR };

enum DrawMode { kDRAW_VERTICES,
                kDRAW_WIREFRAME,
                kDRAW_FACES };


int   g_frame = 0,
      g_repeatCount = 0;

// GUI variables
int   g_fullscreen = 0,
      g_mbutton[3] = {0, 0, 0},
      g_running = 1;

int   g_displayPatchColor    = 1,               
      g_FarDrawMode          = kDRAW_FACES,      
      g_FarDrawVertIDs       = false,           
      g_FarDrawEdgeIDs       = false,           
      g_FarDrawFaceIDs       = false,           
      g_FarDrawPtexIDs       = false,           
      g_FarDrawEdgeSharpness = false,           
      g_FarDrawGregogyBasis  = false,           
      g_FarDrawFVarVerts     = false,           
      g_FarDrawFVarPatches   = false,           
      g_FarDrawFVarPatchTess = 5,               
      g_numPatches           = 0,               
      g_maxValence           = 0,               
      g_currentPatch         = 0,               
      g_Adaptive             = true,
      g_useStencils          = true;

typedef OpenSubdiv::Sdc::Options SdcOptions;

SdcOptions::FVarLinearInterpolation g_fvarInterpolation =
    SdcOptions::FVAR_LINEAR_ALL;

OpenSubdiv::Far::PatchDescriptor g_currentPatchDesc;
OpenSubdiv::Far::PatchDescriptor::Type g_currentFVarPatchType;

float g_rotate[2] = {0, 0},
      g_dolly = 5,
      g_pan[2] = {0, 0},
      g_center[3] = {0, 0, 0},
      g_size = 0;

int   g_prev_x = 0,
      g_prev_y = 0;

int   g_width = 1024,
      g_height = 1024;

GLhud g_hud;

GLFont * g_font=0;

// performance
float g_cpuTime = 0;
float g_gpuTime = 0;
Stopwatch g_fpsTimer;

GLuint g_queries[2] = {0, 0};

GLuint g_transformUB = 0,
       g_lightingUB = 0;

struct Transform {
    float ModelViewMatrix[16];
    float ProjectionMatrix[16];
    float ModelViewProjectionMatrix[16];
} g_transformData;

static GLMesh g_far_glmesh;

static GLControlMeshDisplay g_controlMeshDisplay;
static GLuint g_controlMeshDisplayVBO = 0;


//------------------------------------------------------------------------------

typedef OpenSubdiv::Far::TopologyRefiner               FTopologyRefiner;
typedef OpenSubdiv::Far::TopologyRefinerFactory<Shape> FTopologyRefinerFactory;


//------------------------------------------------------------------------------
// Vertex class implementation
struct Vertex {

    Vertex() { /* _pos[0]=_pos[1]=_pos[2]=0.0f; */ }

    Vertex( int /*i*/ ) { }

    Vertex( float x, float y, float z ) { _pos[0]=x; _pos[1]=y; _pos[2]=z; }

    Vertex( const Vertex & src ) { _pos[0]=src._pos[0]; _pos[1]=src._pos[1]; _pos[2]=src._pos[2]; }

   ~Vertex( ) { }

    void AddWithWeight(Vertex const & src, float weight) {
        _pos[0]+=weight*src._pos[0];
        _pos[1]+=weight*src._pos[1];
        _pos[2]+=weight*src._pos[2];
    }

    void AddWithWeight(Vertex const & src, float weight, float /* ds */, float /* dt */) {
        _pos[0]+=weight*src._pos[0];
        _pos[1]+=weight*src._pos[1];
        _pos[2]+=weight*src._pos[2];
    }

    void Clear( void * =0 ) { _pos[0]=_pos[1]=_pos[2]=0.0f; }

    void SetPosition(float x, float y, float z) { _pos[0]=x; _pos[1]=y; _pos[2]=z; }

    float const * GetPos() const { return _pos; }

private:
    float _pos[3];
};

//------------------------------------------------------------------------------
// generate display IDs for Far verts
static void
createVertNumbers(OpenSubdiv::Far::TopologyRefiner const & refiner,
    std::vector<Vertex> const & vertexBuffer) {

    int maxlevel = refiner.GetMaxLevel(),
        firstvert = 0;

    if (refiner.IsUniform()) {
        for (int i=0; i<maxlevel; ++i) {
            firstvert += refiner.GetLevel(i).GetNumVertices();
        }
    }

    static char buf[16];
    if (refiner.IsUniform()) {
        for (int i=firstvert; i<(int)vertexBuffer.size(); ++i) {
            snprintf(buf, 16, "%d", i);
            g_font->Print3D(vertexBuffer[i].GetPos(), buf, 1);
        }
    } else {

        for (int level=0, vert=0; level<=refiner.GetMaxLevel(); ++level) {
            for (int i=0; i<refiner.GetLevel(level).GetNumVertices(); ++i, ++vert) {
                snprintf(buf, 16, "%d", i);
                g_font->Print3D(vertexBuffer[vert].GetPos(), buf, 1);
            }
        }
    }
}

//------------------------------------------------------------------------------
// generate display IDs for Far edges
static void
createEdgeNumbers(OpenSubdiv::Far::TopologyRefiner const & refiner,
    std::vector<Vertex> const & vertexBuffer, bool ids=false, bool sharpness=false) {

    if (ids || sharpness) {

        int maxlevel = refiner.GetMaxLevel(),
            firstvert = 0;

        for (int i=0; i<maxlevel; ++i) {
            firstvert += refiner.GetLevel(i).GetNumVertices();
        }

        OpenSubdiv::Far::TopologyLevel const & refLastLevel = refiner.GetLevel(maxlevel);

        static char buf[16];
        for (int i=0; i<refLastLevel.GetNumEdges(); ++i) {

            Vertex center(0.0f, 0.0f, 0.0f);

            OpenSubdiv::Far::ConstIndexArray const verts = refLastLevel.GetEdgeVertices(i);
            assert(verts.size()==2);

            center.AddWithWeight(vertexBuffer[firstvert+verts[0]], 0.5f);
            center.AddWithWeight(vertexBuffer[firstvert+verts[1]], 0.5f);

            if (ids) {
                snprintf(buf, 16, "%d", i);
                g_font->Print3D(center.GetPos(), buf, 3);
            }

            if (sharpness) {
                float sharpness = refLastLevel.GetEdgeSharpness(i);
                if (sharpness>0.0f) {
                    snprintf(buf, 16, "%g", sharpness);
                    g_font->Print3D(center.GetPos(), buf, std::min(8,(int)sharpness+4));
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// generate display IDs for Far faces
static void
createFaceNumbers(OpenSubdiv::Far::TopologyRefiner const & refiner,
    std::vector<Vertex> const & vertexBuffer) {

    static char buf[16];

    if (refiner.IsUniform()) {
        int maxlevel = refiner.GetMaxLevel(),
            firstvert = 0;

        for (int i=0; i<maxlevel; ++i) {
            firstvert += refiner.GetLevel(i).GetNumVertices();
        }

        OpenSubdiv::Far::TopologyLevel const & refLastLevel = refiner.GetLevel(maxlevel);

        for (int face=0; face<refLastLevel.GetNumFaces(); ++face) {

            Vertex center(0.0f, 0.0f, 0.0f);

            OpenSubdiv::Far::ConstIndexArray const verts = refLastLevel.GetFaceVertices(face);

            float weight = 1.0f / (float)verts.size();

            for (int vert=0; vert<verts.size(); ++vert) {
                center.AddWithWeight(vertexBuffer[firstvert+verts[vert]], weight);
            }

            snprintf(buf, 16, "%d", face);
            g_font->Print3D(center.GetPos(), buf, 2);
        }
    } else {
        int maxlevel = refiner.GetMaxLevel(),
//            patch = refiner.GetLevel(0).GetNumFaces(),
            firstvert = refiner.GetLevel(0).GetNumVertices();

        for (int level=1; level<=maxlevel; ++level) {

            OpenSubdiv::Far::TopologyLevel const & refLevel = refiner.GetLevel(level);

            int nfaces = refLevel.GetNumFaces();

            for (int face=0; face<nfaces; ++face /*, ++patch */) {

                Vertex center(0.0f, 0.0f, 0.0f);

                OpenSubdiv::Far::ConstIndexArray const verts = refLevel.GetFaceVertices(face);

                float weight = 1.0f / (float)verts.size();

                for (int vert=0; vert<verts.size(); ++vert) {
                    center.AddWithWeight(vertexBuffer[firstvert+verts[vert]], weight);
                }
                snprintf(buf, 16, "%d", face);
                g_font->Print3D(center.GetPos(), buf, 2);
            }
            firstvert+=refLevel.GetNumVertices();
        }
    }
}

//------------------------------------------------------------------------------
// generate display vert IDs for the selected Far patch
static void
createPatchNumbers(OpenSubdiv::Far::PatchTable const & patchTable,
    std::vector<Vertex> const & vertexBuffer) {

    if (! g_currentPatch)
        return;

    int patchID = g_currentPatch-1,
        patchArray = -1;

    // Find PatchArray containing our patch
    for (int array=0; array<(int)patchTable.GetNumPatchArrays(); ++array) {
        int npatches = patchTable.GetNumPatches(array);
        if (patchID >= npatches) {
            patchID -= npatches;
        } else {
            patchArray = array;
            break;
        }
    }
    if (patchArray==-1) {
        return;
    }

    g_currentPatchDesc = patchTable.GetPatchArrayDescriptor(patchArray);

    OpenSubdiv::Far::ConstIndexArray const cvs =
        patchTable.GetPatchVertices(patchArray, patchID);

    static char buf[16];
    for (int i=0; i<cvs.size(); ++i) {
        snprintf(buf, 16, "%d", i);
        g_font->Print3D(vertexBuffer[cvs[i]].GetPos(), buf, 1);
    }
}

//------------------------------------------------------------------------------
// generate display vert IDs for the selected Far FVar patch
static void
createFVarPatchNumbers(OpenSubdiv::Far::PatchTable const & patchTable,
    std::vector<Vertex> const & fvarBuffer) {

    static int channel = 0;

    int patch = g_currentPatch-1;
    static char buf[16];

    if (patch>=0 && patch<patchTable.GetNumPatchesTotal()) {

        OpenSubdiv::Far::PatchTable::PatchHandle handle;
        handle.patchIndex = patch;

        OpenSubdiv::Far::ConstIndexArray const cvs =
            patchTable.GetPatchFVarValues(handle, channel);

        for (int i=0; i<cvs.size(); ++i) {
            snprintf(buf, 16, "%d", i);
            g_font->Print3D(fvarBuffer[cvs[i]].GetPos(), buf, 2);
        }

    }
}

//------------------------------------------------------------------------------
// generate display for Far FVar patches
static GLMesh fvarVerts,
              fvarWire;

static void
createFVarPatches(OpenSubdiv::Far::TopologyRefiner const & refiner,
    OpenSubdiv::Far::PatchTable const & patchTable,
        std::vector<Vertex> const & fvarBuffer) {

    assert(!fvarBuffer.empty());

    static int channel = 0;

    if (g_FarDrawFVarVerts) {
        GLMesh::Options options;
        options.vertColorMode = GLMesh::VERTCOLOR_BY_LEVEL;
        fvarVerts.InitializeFVar(options, refiner, &patchTable, channel, 0, (float *)(&fvarBuffer[0]));
    }

    if (g_FarDrawFVarPatches) {

        // generate uniform tessellation for patches
        int tessFactor = g_FarDrawFVarPatchTess,
            npatches = patchTable.GetNumPatchesTotal(),
            nvertsperpatch = (tessFactor) * (tessFactor),
            nverts = npatches * nvertsperpatch;

        float * uvs = (float *)alloca(tessFactor);
        for (int i=0; i<tessFactor; ++i) {
            uvs[i] = (float)i/(tessFactor-1.0f);
        }

        std::vector<Vertex> verts(nverts);
        memset(&verts[0], 0, verts.size()*sizeof(Vertex));

        /*
        OpenSubdiv::Far::PatchTable::PatchHandle handle;

        Vertex * vert = &verts[0];
        for (int patch=0; patch<npatches; ++patch) {
            for (int i=0; i<tessFactor; ++i) {
                for (int j=0; j<tessFactor; ++j, ++vert) {
                    handle.patchIndex = patch;
                    //  To be replaced with EvaluateBasis() for the appropriate channel:
                    //patchTable.EvaluateFaceVarying(channel, handle, uvs[i], uvs[j], fvarBuffer, *vert);
                }
            }
        }
        */

        GLMesh::Options options;
        options.edgeColorMode = GLMesh::EDGECOLOR_BY_PATCHTYPE;
        fvarWire.InitializeFVar(options, refiner, &patchTable, channel, tessFactor, (float *)(&verts[0]));
    }
}

//------------------------------------------------------------------------------
// generate display IDs for Far Gregory basis

static GLMesh gregoryWire;

static void
createGregoryBasis(OpenSubdiv::Far::PatchTable const & patchTable,
        std::vector<Vertex> const & vertexBuffer) {

    typedef OpenSubdiv::Far::PatchDescriptor PatchDescriptor;

    int npatches = 0;
    int patchArray = 0;
    for (int array=0; array<(int)patchTable.GetNumPatchArrays(); ++array) {
        if (patchTable.GetPatchArrayDescriptor(array).GetType()==
            PatchDescriptor::GREGORY_BASIS) {
            npatches = patchTable.GetNumPatches(array);
            patchArray = array;
            break;
        }
    }

    int nedges = npatches * 20;
    std::vector<int> vertsperedge(nedges), edgeindices(nedges*2);

    for (int patch=0; patch<npatches; ++patch) {

        static int  basisedges[40] = {  0,  1,  0,  2,  1,  3,  2,  4,
                                        5,  6,  5,  7,  6,  8,  7,  9,
                                       10, 11, 10, 12, 11, 13, 12, 14,
                                       15, 16, 15, 17, 16, 18, 17, 19,
                                        1,  7,  6, 12, 11, 17, 16,  2  };

        int offset = patch * 20,
            * vpe = &vertsperedge[offset],
            * indices = &edgeindices[patch * 40];

        OpenSubdiv::Far::ConstIndexArray const cvs =
            patchTable.GetPatchVertices(patchArray, patch);

        for (int i=0; i<20; ++i) {
            vpe[i] = 2;
            indices[i*2] = cvs[basisedges[i*2]];
            indices[i*2+1] = cvs[basisedges[i*2+1]];
        }

        //Vertex const * verts = &edgeverts[offset];
        static char buf[16];
        for (int i=0; i<4; ++i) {
            int vid = patch * 20 + i * 5;

            const float *P  = vertexBuffer[cvs[i*5+0]].GetPos();
            const float *Ep = vertexBuffer[cvs[i*5+1]].GetPos();
            const float *Em = vertexBuffer[cvs[i*5+2]].GetPos();
            const float *Fp = vertexBuffer[cvs[i*5+3]].GetPos();
            const float *Fm = vertexBuffer[cvs[i*5+4]].GetPos();

            snprintf(buf, 16, " P%d (%d)", i, vid);
            g_font->Print3D(P, buf, 3);
            snprintf(buf, 16, " Ep%d (%d)", i, vid+1);
            g_font->Print3D(Ep, buf, 3);
            snprintf(buf, 16, " Em%d (%d)", i, vid+2);
            g_font->Print3D(Em, buf, 3);
            snprintf(buf, 16, " Fp%d (%d)", i, vid+3);
            g_font->Print3D(Fp, buf, 3);
            snprintf(buf, 16, " Fm%d (%d)", i, vid+4);
            g_font->Print3D(Fm, buf, 3);
        }
    }

    GLMesh::Options options;
    gregoryWire.Initialize(options, (int)vertexBuffer.size(), (int)vertsperedge.size(),
                           &vertsperedge[0], &edgeindices[0], (float const *)&vertexBuffer[0]);
}

//------------------------------------------------------------------------------
// generate display IDs for Far faces
static void
createPtexNumbers(OpenSubdiv::Far::PatchTable const & patchTable,
    std::vector<Vertex> const & vertexBuffer) {

    typedef OpenSubdiv::Far::PatchDescriptor Descriptor;

    static char buf[16];

    static int regular[4]  = {5, 6, 9, 10},
               gregory[4]  = {0, 1, 2, 3};

    for (int array=0; array<(int)patchTable.GetNumPatchArrays(); ++array) {

        for (int patch=0; patch<(int)patchTable.GetNumPatches(array); ++patch) {

            OpenSubdiv::Far::ConstIndexArray const cvs =
                patchTable.GetPatchVertices(array, patch);

            int * remap = 0;
            switch (patchTable.GetPatchArrayDescriptor(array).GetType()) {
                case Descriptor::REGULAR:          remap = regular; break;
                case Descriptor::GREGORY:
                case Descriptor::GREGORY_BOUNDARY:
                case Descriptor::GREGORY_BASIS:    remap = gregory; break;
                default:
                    assert(0);
            }

            Vertex center(0.0f, 0.0f, 0.0f);
            for (int k=0; k<4; ++k) {
                center.AddWithWeight(vertexBuffer[cvs[remap[k]]], 0.25f);
            }

            snprintf(buf, 16, "%d", patchTable.GetPatchParam(array, patch).GetFaceId());
            g_font->Print3D(center.GetPos(), buf, 1);
        }
    }
}

//------------------------------------------------------------------------------
static void
createFarGLMesh(Shape * shape, int maxlevel) {

    Stopwatch s;
    s.Start();

    using namespace OpenSubdiv;

    // create Far mesh (topology)
    Sdc::SchemeType sdctype = GetSdcType(*shape);
    Sdc::Options    sdcoptions = GetSdcOptions(*shape);

    sdcoptions.SetFVarLinearInterpolation(g_fvarInterpolation);

    Far::TopologyRefiner * refiner =
        Far::TopologyRefinerFactory<Shape>::Create(*shape,
            Far::TopologyRefinerFactory<Shape>::Options(sdctype, sdcoptions));

    if (g_Adaptive) {
        Far::TopologyRefiner::AdaptiveOptions options(maxlevel);
        options.useSingleCreasePatch = false;
        refiner->RefineAdaptive(options);
    } else {
        Far::TopologyRefiner::UniformOptions options(maxlevel);
        options.fullTopologyInLastLevel = true;
        refiner->RefineUniform(options);
    }

    int numTotalVerts = refiner->GetNumVerticesTotal();

    //
    // Patch table
    //
    std::vector<Vertex> fvarBuffer;
    Far::PatchTable * patchTable = 0;
    bool createFVarWire = g_FarDrawFVarPatches || g_FarDrawFVarVerts;

    if (g_Adaptive) {
        Far::PatchTableFactory::Options options;
        options.generateFVarTables = createFVarWire;
        options.shareEndCapPatchPoints = false;

        patchTable =
            Far::PatchTableFactory::Create(*refiner, options);

        // increase vertex buffer for the additional local points
        if (patchTable->GetLocalPointStencilTable()) {
            numTotalVerts += patchTable->GetLocalPointStencilTable()->GetNumStencils();
        }

        g_numPatches = patchTable->GetNumPatchesTotal();
        g_maxValence = patchTable->GetMaxValence();

        if (createFVarWire) {

            // interpolate fvar values

            int channel = 0;

            // XXXX should use a (u,v) vertex class
            fvarBuffer.resize(refiner->GetNumFVarValuesTotal(channel), 0);
            Vertex * values = &fvarBuffer[0];

            int nCoarseValues = refiner->GetLevel(0).GetNumFVarValues(channel);

            for (int i=0; i<nCoarseValues; ++i) {
                float const * ptr = &shape->uvs[i*2];
                values[i].SetPosition(ptr[0],  ptr[1], 0.0f);
            }

            int lastLevel = refiner->GetMaxLevel();
            Vertex * src = values;
            for (int level = 1; level <= lastLevel; ++level) {
                Vertex * dst = src + refiner->GetLevel(level-1).GetNumFVarValues(channel);
                Far::PrimvarRefiner(*refiner).InterpolateFaceVarying(level, src, dst, channel);
                src = dst;
            }
        }
    }

    //
    // interpolate vertices
    //

    // create vertex primvar data buffer
    std::vector<Vertex> vertexBuffer(numTotalVerts);
    Vertex * verts = &vertexBuffer[0];

    // copy coarse vertices positions
    int ncoarseverts = shape->GetNumVertices();
    for (int i=0; i<ncoarseverts; ++i) {
        float * ptr = &shape->verts[i*3];
        verts[i].SetPosition(ptr[0], ptr[1], ptr[2]);
    }

    s.Start();
    if (g_useStencils) {
        //
        // Stencil interpolation
        //
        Far::StencilTable const * stencilTable = 0;
        Far::StencilTableFactory::Options options;
        options.generateOffsets=true;
        options.generateIntermediateLevels=true;
        stencilTable = Far::StencilTableFactory::Create(*refiner, options);

        // append local point stencils if needed
        if (patchTable && patchTable->GetLocalPointStencilTable()) {
            if (Far::StencilTable const * stencilTableWithLocalPoints =
                Far::StencilTableFactory::AppendLocalPointStencilTable(
                    *refiner, stencilTable,
                    patchTable->GetLocalPointStencilTable())) {
                delete stencilTable;
                stencilTable = stencilTableWithLocalPoints;
            }
        }

        //
        // apply stencils
        //
        stencilTable->UpdateValues(verts, verts + ncoarseverts);

        delete stencilTable;
    } else {
        //
        // TopologyRefiner interpolation
        //
        // populate buffer with Far interpolated vertex data
        int lastLevel = refiner->GetMaxLevel();
        Vertex * src = verts;
        for (int level = 1; level <= lastLevel; ++level) {
            Vertex * dst = src + refiner->GetLevel(level-1).GetNumVertices();
            Far::PrimvarRefiner(*refiner).Interpolate(level, src, dst);
            src = dst;
        }
        //printf("          %f ms (interpolate)\n", float(s.GetElapsed())*1000.0f);
        //printf("          %f ms (total)\n", float(s.GetTotalElapsed())*1000.0f);

        // TODO: endpatch basis conversion comes here
    }
    s.Stop();

    //
    // Misc display
    //

    //printf("Far time: %f ms (topology)\n", float(s.GetElapsed())*1000.0f);

    if (g_FarDrawVertIDs) {
        createVertNumbers(*refiner, vertexBuffer);
    }

    if (g_FarDrawFaceIDs) {
        createFaceNumbers(*refiner, vertexBuffer);
    }

    if (g_FarDrawPtexIDs && patchTable) {
        createPtexNumbers(*patchTable, vertexBuffer);
    }

    if (g_Adaptive) {
        createPatchNumbers(*patchTable, vertexBuffer);
    }

    if (g_Adaptive && g_FarDrawGregogyBasis) {
        createGregoryBasis(*patchTable, vertexBuffer);
    }

    if (g_Adaptive && createFVarWire) {
        createFVarPatches(*refiner, *patchTable, fvarBuffer);
        createFVarPatchNumbers(*patchTable, fvarBuffer);
    }

    createEdgeNumbers(*refiner, vertexBuffer, g_FarDrawEdgeIDs!=0, g_FarDrawEdgeSharpness!=0);

    GLMesh::Options options;
    options.vertColorMode=g_Adaptive ? GLMesh::VERTCOLOR_BY_LEVEL : GLMesh::VERTCOLOR_BY_SHARPNESS;
    options.edgeColorMode=g_Adaptive ? GLMesh::EDGECOLOR_BY_PATCHTYPE : GLMesh::EDGECOLOR_BY_SHARPNESS;
    options.faceColorMode=g_Adaptive ? GLMesh::FACECOLOR_BY_PATCHTYPE :GLMesh::FACECOLOR_SOLID;

    g_far_glmesh.Initialize(options, *refiner, patchTable, (float *)&verts[0]);
    if (g_Adaptive) {
        g_far_glmesh.SetDiffuseColor(1.0f, 1.0f, 1.0f, 1.0f);
    } else {
        g_far_glmesh.SetDiffuseColor(0.75f, 0.9f, 1.0f, 1.0f);
    }


    //setFaceColors(*refiner);

    g_far_glmesh.InitializeDeviceBuffers();

    // save coarse topology (used for control mesh display)
    g_controlMeshDisplay.SetTopology(refiner->GetLevel(0));

    // save coarse points in a GPU buffer (used for control mesh display)
    if (! g_controlMeshDisplayVBO) {
        glGenBuffers(1, &g_controlMeshDisplayVBO);
    }
    glBindBuffer(GL_ARRAY_BUFFER, g_controlMeshDisplayVBO);
    glBufferData(GL_ARRAY_BUFFER,
                 3*sizeof(float)*vertexBuffer.size(), (GLfloat*)&vertexBuffer[0],
                 GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // compute model bounds
    float min[3] = { FLT_MAX,  FLT_MAX,  FLT_MAX};
    float max[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (size_t i=0; i <vertexBuffer.size(); ++i) {
        for(int j=0; j<3; ++j) {
            float v = vertexBuffer[i].GetPos()[j];
            min[j] = std::min(min[j], v);
            max[j] = std::max(max[j], v);
        }
    }
    for (int j=0; j<3; ++j) {
        g_center[j] = (min[j] + max[j]) * 0.5f;
        g_size += (max[j]-min[j])*(max[j]-min[j]);
    }
    g_size = sqrtf(g_size);

    delete refiner;
    delete patchTable;
}

//------------------------------------------------------------------------------
static void
createMeshes(ShapeDesc const & desc, int maxlevel) {

    if (! g_font) {
        g_font = new GLFont(g_hud.GetFontTexture());
    }
    g_font->Clear();

    Shape * shape = Shape::parseObj(desc);

    createFarGLMesh(shape, maxlevel);
    delete shape;
}

//------------------------------------------------------------------------------
static void
fitFrame() {

    g_pan[0] = g_pan[1] = 0;
    g_dolly = g_size;
}

//------------------------------------------------------------------------------
static void
display() {

    Stopwatch s;
    s.Start();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glViewport(0, 0, g_width, g_height);

    // prepare view matrix
    double aspect = g_width/(double)g_height;
    identity(g_transformData.ModelViewMatrix);
    translate(g_transformData.ModelViewMatrix, -g_pan[0], -g_pan[1], -g_dolly);
    rotate(g_transformData.ModelViewMatrix, g_rotate[1], 1, 0, 0);
    rotate(g_transformData.ModelViewMatrix, g_rotate[0], 0, 1, 0);
    rotate(g_transformData.ModelViewMatrix, -90, 1, 0, 0);
    translate(g_transformData.ModelViewMatrix,
              -g_center[0], -g_center[1], -g_center[2]);
    perspective(g_transformData.ProjectionMatrix,
                45.0f, (float)aspect, 0.1f, 500.0f);
    multMatrix(g_transformData.ModelViewProjectionMatrix,
               g_transformData.ModelViewMatrix,
               g_transformData.ProjectionMatrix);

    glEnable(GL_DEPTH_TEST);

    s.Stop();
    float drawCpuTime = float(s.GetElapsed() * 1000.0f);

    glBindVertexArray(0);

    glUseProgram(0);

    // primitive counting

    glBeginQuery(GL_PRIMITIVES_GENERATED, g_queries[0]);
#if defined(GL_VERSION_3_3)
    glBeginQuery(GL_TIME_ELAPSED, g_queries[1]);
#endif

    // Update and bind transform state ---------------------
    if (! g_transformUB) {
        glGenBuffers(1, &g_transformUB);
        glBindBuffer(GL_UNIFORM_BUFFER, g_transformUB);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(g_transformData), NULL, GL_STATIC_DRAW);
    };
    glBindBuffer(GL_UNIFORM_BUFFER, g_transformUB);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(g_transformData), &g_transformData);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    // Update and bind lighting state ----------------------
    struct Lighting {
        struct Light {
            float position[4];
            float ambient[4];
            float diffuse[4];
            float specular[4];
        } lightSource[2];
    } lightingData = {
       {{  { 0.5,  0.2f, 1.0f, 0.0f },
           { 0.1f, 0.1f, 0.1f, 1.0f },
           { 0.7f, 0.7f, 0.7f, 1.0f },
           { 0.8f, 0.8f, 0.8f, 1.0f } },

         { { -0.8f, 0.4f, -1.0f, 0.0f },
           {  0.0f, 0.0f,  0.0f, 1.0f },
           {  0.5f, 0.5f,  0.5f, 1.0f },
           {  0.8f, 0.8f,  0.8f, 1.0f } }}
    };
    if (! g_lightingUB) {
        glGenBuffers(1, &g_lightingUB);
        glBindBuffer(GL_UNIFORM_BUFFER, g_lightingUB);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(lightingData), NULL, GL_STATIC_DRAW);
    };
    glBindBuffer(GL_UNIFORM_BUFFER, g_lightingUB);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(lightingData), &lightingData);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    // Draw stuff ------------------------------------------

    // control mesh edges & verts
    g_controlMeshDisplay.Draw(g_controlMeshDisplayVBO, 3*sizeof(float),
                              g_transformData.ModelViewProjectionMatrix);

    // Far mesh
    GLMesh::Component comp=GLMesh::COMP_VERT;
    switch (g_FarDrawMode) {
        case kDRAW_VERTICES  : comp=GLMesh::COMP_VERT; break;
        case kDRAW_WIREFRAME : comp=GLMesh::COMP_EDGE; break;
        case kDRAW_FACES     : comp=GLMesh::COMP_FACE; break;
        default:
            assert(0);
    }
    g_far_glmesh.Draw(comp, g_transformUB, g_lightingUB);

    if (g_Adaptive && g_FarDrawGregogyBasis) {
        gregoryWire.Draw(GLMesh::COMP_VERT, g_transformUB, g_lightingUB);
        gregoryWire.Draw(GLMesh::COMP_EDGE, g_transformUB, g_lightingUB);
    }

    if (g_Adaptive && g_FarDrawFVarVerts) {
        fvarVerts.Draw(GLMesh::COMP_VERT, g_transformUB, g_lightingUB);
    }
    if (g_Adaptive && g_FarDrawFVarPatches) {
        fvarWire.Draw(GLMesh::COMP_EDGE, g_transformUB, g_lightingUB);
    }

    assert(g_font);
    g_font->Draw(g_transformUB);

    // -----------------------------------------------------

    glEndQuery(GL_PRIMITIVES_GENERATED);
#if defined(GL_VERSION_3_3)
    glEndQuery(GL_TIME_ELAPSED);
#endif

    GLuint numPrimsGenerated = 0;
    GLuint timeElapsed = 0;
    glGetQueryObjectuiv(g_queries[0], GL_QUERY_RESULT, &numPrimsGenerated);
#if defined(GL_VERSION_3_3)
    glGetQueryObjectuiv(g_queries[1], GL_QUERY_RESULT, &timeElapsed);
#endif

    float drawGpuTime = timeElapsed / 1000.0f / 1000.0f;

    if (g_hud.IsVisible()) {
        g_fpsTimer.Stop();
        double fps = 1.0/g_fpsTimer.GetElapsed();
        g_fpsTimer.Start();

        { // display selected patch info
            static char const * patchTypes[13] = { "undefined", "points", "lines",
                "quads", "tris", "loop", "regular", "single crease", "boundary", "corner",
                    "gregory", "gregory-boundary", "gregory-basis" },
                              * format0 = "Current Patch : %d/%d (%s - %d CVs)",
                              * format1 = "Current Patch : %d/%d (%s - %d CVs) fvar: (%s - %d CVs)";

            if (g_Adaptive && g_currentPatch) {
                
                if (g_FarDrawFVarPatches || g_FarDrawFVarVerts) {

                    g_hud.DrawString(g_width/2-200, 225, format1,
                        g_currentPatch-1, g_numPatches-1,
                            patchTypes[g_currentPatchDesc.GetType()],
                                g_currentPatchDesc.GetNumControlVertices(),
                                    patchTypes[g_currentFVarPatchType],
                                        OpenSubdiv::Far::PatchDescriptor::GetNumFVarControlVertices(
                                            g_currentFVarPatchType));
                } else {
                    g_hud.DrawString(g_width/2-200, 225, format0,
                        g_currentPatch-1, g_numPatches-1,
                            patchTypes[g_currentPatchDesc.GetType()],
                                g_currentPatchDesc.GetNumControlVertices(),
                                    patchTypes[g_currentFVarPatchType]);
                }
            }
        }

        static char const * schemeNames[3] = { "BILINEAR", "CATMARK", "LOOP" };

        g_hud.DrawString(10, -140, "Primitives : %d", numPrimsGenerated);
        g_hud.DrawString(10, -120, "Scheme     : %s", schemeNames[ g_shapes[g_currentShape].scheme ]);
        g_hud.DrawString(10, -100, "GPU Kernel : %.3f ms", g_gpuTime);
        g_hud.DrawString(10, -80,  "CPU Kernel : %.3f ms", g_cpuTime);
        g_hud.DrawString(10, -60,  "GPU Draw   : %.3f ms", drawGpuTime);
        g_hud.DrawString(10, -40,  "CPU Draw   : %.3f ms", drawCpuTime);
        g_hud.DrawString(10, -20,  "FPS        : %3.1f", fps);

        g_hud.Flush();
    }
    glFinish();

    //checkGLErrors("display leave");
}

//------------------------------------------------------------------------------
static void
motion(GLFWwindow *, double dx, double dy) {
    int x=(int)dx, y=(int)dy;

    if (g_hud.MouseCapture()) {
        // check gui
        g_hud.MouseMotion(x, y);
    } else if (g_mbutton[0] && !g_mbutton[1] && !g_mbutton[2]) {
        // orbit
        g_rotate[0] += x - g_prev_x;
        g_rotate[1] += y - g_prev_y;
    } else if (!g_mbutton[0] && !g_mbutton[1] && g_mbutton[2]) {
        // pan
        g_pan[0] -= g_dolly*(x - g_prev_x)/g_width;
        g_pan[1] += g_dolly*(y - g_prev_y)/g_height;
    } else if ((g_mbutton[0] && !g_mbutton[1] && g_mbutton[2]) ||
               (!g_mbutton[0] && g_mbutton[1] && !g_mbutton[2])) {
        // dolly
        g_dolly -= g_dolly*0.01f*(x - g_prev_x);
        if(g_dolly <= 0.01) g_dolly = 0.01f;
    }

    g_prev_x = x;
    g_prev_y = y;
}

//------------------------------------------------------------------------------
static void
mouse(GLFWwindow *, int button, int state, int /* mods */) {

    if (state == GLFW_RELEASE)
        g_hud.MouseRelease();

    if (button == 0 && state == GLFW_PRESS && g_hud.MouseClick(g_prev_x, g_prev_y))
        return;

    if (button < 3) {
        g_mbutton[button] = (state == GLFW_PRESS);
    }
}

//------------------------------------------------------------------------------
static void
reshape(GLFWwindow *, int width, int height) {

    g_width = width;
    g_height = height;

    int windowWidth = g_width, windowHeight = g_height;

    // window size might not match framebuffer size on a high DPI display
    glfwGetWindowSize(g_window, &windowWidth, &windowHeight);

    g_hud.Rebuild(windowWidth, windowHeight, width, height);
}

//------------------------------------------------------------------------------
void windowClose(GLFWwindow*) {
    g_running = false;
}

//------------------------------------------------------------------------------
static void
toggleFullScreen() {
    // XXXX manuelk : to re-implement from glut
}

//------------------------------------------------------------------------------
static void
rebuildOsdMeshes() {

    createMeshes(g_shapes[ g_currentShape ], g_level);
}

//------------------------------------------------------------------------------
static void
keyboard(GLFWwindow *, int key, int /* scancode */, int event, int /* mods */) {

    if (event == GLFW_RELEASE) return;
    if (g_hud.KeyDown(tolower(key))) return;

    switch (key) {
        case 'Q': g_running = 0; break;
        case 'F': fitFrame(); break;

        case '[': if (g_currentPatch > 0) {
                      --g_currentPatch;
                      rebuildOsdMeshes();
                  } break;

        case ']': if (g_currentPatch < g_numPatches) {
                      ++g_currentPatch;
                      rebuildOsdMeshes();
                  } break;

        case GLFW_KEY_TAB: toggleFullScreen(); break;
        case GLFW_KEY_ESCAPE: g_hud.SetVisible(!g_hud.IsVisible()); break;
    }
}

static void
callbackLevel(int l) {

    g_level = l;
    rebuildOsdMeshes();
}

static void
callbackModel(int m) {

    if (m < 0)
        m = 0;

    if (m >= (int)g_shapes.size())
        m = (int)g_shapes.size() - 1;

    g_currentShape = m;
    rebuildOsdMeshes();
}

static void
callbackAdaptive(bool checked, int /* a */)
{
    g_Adaptive = checked;
    rebuildOsdMeshes();
}

static void
callbackUseStencils(bool checked, int /* a */)
{
    g_useStencils = checked;
    rebuildOsdMeshes();
}

static void
callbackCheckBox(bool checked, int button) {

    switch (button) {
        case kHUD_CB_DISPLAY_CONTROL_MESH_EDGES :
            g_controlMeshDisplay.SetEdgesDisplay(checked);
            break;
        case kHUD_CB_DISPLAY_CONTROL_MESH_VERTS :
            g_controlMeshDisplay.SetVerticesDisplay(checked);
            break;
        case kHUD_CB_DISPLAY_PATCH_COLOR:
            g_displayPatchColor = checked;
            break;
    }
}


static void
callbackFarDrawMode(int m) {

    g_FarDrawMode = m;
}

static void
callbackDrawIDs(bool checked, int button) {

    switch (button) {

        case 0: g_FarDrawVertIDs = checked; break;
        case 1: g_FarDrawEdgeIDs = checked; break;
        case 2: g_FarDrawFaceIDs = checked; break;
        case 3: g_FarDrawPtexIDs = checked; break;
        case 4: g_FarDrawEdgeSharpness = checked; break;
        case 5: g_FarDrawGregogyBasis = checked; break;

        case 6: g_FarDrawFVarVerts = checked; break;
        case 7: g_FarDrawFVarPatches = checked; break;

        default: break;
    }
    rebuildOsdMeshes();
}

static void
callbackScale(float value, int) {

    g_font->SetFontScale(value);
}

static void
callbackFVarTess(float value, int) {

    g_FarDrawFVarPatchTess = (int)value;
    rebuildOsdMeshes();
}

static void
callbackFVarInterpolation(int b) {

    switch (b) {

        case SdcOptions::FVAR_LINEAR_NONE :
            g_fvarInterpolation = SdcOptions::FVAR_LINEAR_NONE; break;

        case SdcOptions::FVAR_LINEAR_CORNERS_ONLY :
            g_fvarInterpolation = SdcOptions::FVAR_LINEAR_CORNERS_ONLY; break;

        case SdcOptions::FVAR_LINEAR_CORNERS_PLUS1 :
            g_fvarInterpolation = SdcOptions::FVAR_LINEAR_CORNERS_PLUS1; break;

        case SdcOptions::FVAR_LINEAR_CORNERS_PLUS2 :
            g_fvarInterpolation = SdcOptions::FVAR_LINEAR_CORNERS_PLUS2; break;

        case SdcOptions::FVAR_LINEAR_BOUNDARIES :
            g_fvarInterpolation = SdcOptions::FVAR_LINEAR_BOUNDARIES; break;

        case SdcOptions::FVAR_LINEAR_ALL :
            g_fvarInterpolation = SdcOptions::FVAR_LINEAR_ALL; break;

    }
    rebuildOsdMeshes();
}

//------------------------------------------------------------------------------
static void
initHUD() {

    int windowWidth = g_width, windowHeight = g_height;
    int frameBufferWidth = g_width, frameBufferHeight = g_height;

    // window size might not match framebuffer size on a high DPI display
    glfwGetWindowSize(g_window, &windowWidth, &windowHeight);
    glfwGetFramebufferSize(g_window, &frameBufferWidth, &frameBufferHeight);

    g_hud.Init(windowWidth, windowHeight, frameBufferWidth, frameBufferHeight);

    g_hud.AddCheckBox("Control edges (H)", g_controlMeshDisplay.GetEdgesDisplay(),
                      10, 10, callbackCheckBox, kHUD_CB_DISPLAY_CONTROL_MESH_EDGES, 'h');
    g_hud.AddCheckBox("Control vertices (J)", g_controlMeshDisplay.GetVerticesDisplay(),
                      10, 30, callbackCheckBox, kHUD_CB_DISPLAY_CONTROL_MESH_VERTS, 'j');


    int pulldown = g_hud.AddPullDown("Far Draw Mode (W)", 10, 195, 250, callbackFarDrawMode, 'w');
    g_hud.AddPullDownButton(pulldown, "Vertices",  0, g_FarDrawMode==kDRAW_VERTICES);
    g_hud.AddPullDownButton(pulldown, "Wireframe", 1, g_FarDrawMode==kDRAW_WIREFRAME);
    g_hud.AddPullDownButton(pulldown, "Faces",     2, g_FarDrawMode==kDRAW_FACES);

    g_hud.AddCheckBox("Vert IDs",   g_FarDrawVertIDs!=0, 10, 215, callbackDrawIDs, 0);
    g_hud.AddCheckBox("Edge IDs",   g_FarDrawEdgeIDs!=0, 10, 235, callbackDrawIDs, 1);
    g_hud.AddCheckBox("Face IDs",   g_FarDrawFaceIDs!=0, 10, 255, callbackDrawIDs, 2);
    g_hud.AddCheckBox("Ptex IDs",   g_FarDrawPtexIDs!=0, 10, 275, callbackDrawIDs, 3);
    g_hud.AddCheckBox("Edge Sharp", g_FarDrawEdgeSharpness!=0, 10, 295, callbackDrawIDs, 4);
    g_hud.AddCheckBox("Gregory Basis", g_FarDrawGregogyBasis!=0, 10, 315, callbackDrawIDs, 5);

    g_hud.AddCheckBox("Use Stencils (S)", g_useStencils!=0, 10, 350, callbackUseStencils, 0, 's');
    g_hud.AddCheckBox("Adaptive (`)", g_Adaptive!=0, 10, 370, callbackAdaptive, 0, '`');


    g_hud.AddSlider("Font Scale", 0.0f, 0.1f, 0.01f,
                    -900, -50, 100, false, callbackScale, 0);

    for (int i = 1; i < 11; ++i) {
        char level[16];
        sprintf(level, "Lv. %d", i);
        g_hud.AddRadioButton(3, level, i==g_level, 10, 380+i*20, callbackLevel, i, '0'+(i%10));
    }

    int shapes_pulldown = g_hud.AddPullDown("Shape (N)", -300, 10, 300, callbackModel, 'n');
    for (int i = 0; i < (int)g_shapes.size(); ++i) {
        g_hud.AddPullDownButton(shapes_pulldown, g_shapes[i].name.c_str(),i, (g_currentShape==i));
    }

   g_hud.AddCheckBox("FVar Verts",  g_FarDrawFVarVerts!=0, 300, 10, callbackDrawIDs, 10);
   g_hud.AddCheckBox("FVar Patches",  g_FarDrawFVarPatches!=0, 300, 30, callbackDrawIDs, 11);

   g_hud.AddSlider("FVar Tess", 1.0f, 10.0f, (float)g_FarDrawFVarPatchTess,
                    300, 50, 25, true, callbackFVarTess, 0);

    int fvar_pulldown = g_hud.AddPullDown("FVar Interpolation (i)", 300, 90, 250, callbackFVarInterpolation, 'i');
    g_hud.AddPullDownButton(fvar_pulldown, "FVAR_LINEAR_NONE",
        SdcOptions::FVAR_LINEAR_NONE, g_fvarInterpolation==SdcOptions::FVAR_LINEAR_NONE);
    g_hud.AddPullDownButton(fvar_pulldown, "FVAR_LINEAR_CORNERS_ONLY",
        SdcOptions::FVAR_LINEAR_CORNERS_ONLY, g_fvarInterpolation==SdcOptions::FVAR_LINEAR_CORNERS_ONLY);
    g_hud.AddPullDownButton(fvar_pulldown, "FVAR_LINEAR_CORNERS_PLUS1",
        SdcOptions::FVAR_LINEAR_CORNERS_PLUS1, g_fvarInterpolation==SdcOptions::FVAR_LINEAR_CORNERS_PLUS1);
    g_hud.AddPullDownButton(fvar_pulldown, "FVAR_LINEAR_CORNERS_PLUS2",
        SdcOptions::FVAR_LINEAR_CORNERS_PLUS2, g_fvarInterpolation==SdcOptions::FVAR_LINEAR_CORNERS_PLUS2);
    g_hud.AddPullDownButton(fvar_pulldown, "FVAR_LINEAR_BOUNDARIES",
        SdcOptions::FVAR_LINEAR_BOUNDARIES, g_fvarInterpolation==SdcOptions::FVAR_LINEAR_BOUNDARIES);
    g_hud.AddPullDownButton(fvar_pulldown, "FVAR_LINEAR_ALL",
        SdcOptions::FVAR_LINEAR_ALL, g_fvarInterpolation==SdcOptions::FVAR_LINEAR_ALL);

    if (! g_font) {
        g_font = new GLFont( g_hud.GetFontTexture() );
    }
}

//------------------------------------------------------------------------------
static void
initGL() {

    glClearColor(0.1f, 0.1f, 0.1f, 0.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);

    glGenQueries(2, g_queries);
}

//------------------------------------------------------------------------------
static void
uninitGL() {

    glDeleteQueries(2, g_queries);
}

//------------------------------------------------------------------------------
static void
idle() {

    if (g_repeatCount != 0 && g_frame >= g_repeatCount)
        g_running = 0;
}

//------------------------------------------------------------------------------
int main(int argc, char ** argv)
{
    bool fullscreen = false;
    std::string str;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-d"))
            g_level = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c"))
            g_repeatCount = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-f"))
            fullscreen = true;
        else {
            std::ifstream ifs(argv[1]);
            if (ifs) {
                std::stringstream ss;
                ss << ifs.rdbuf();
                ifs.close();
                str = ss.str();
                g_shapes.push_back(ShapeDesc(argv[1], str.c_str(), kCatmark));
            }
        }
    }
    initShapes();

    if (! glfwInit()) {
        printf("Failed to initialize GLFW\n");
        return 1;
    }

    static const char windowTitle[] = "OpenSubdiv farViewer";

    GLUtils::SetMinimumGLVersion();

    if (fullscreen) {

        g_primary = glfwGetPrimaryMonitor();

        // apparently glfwGetPrimaryMonitor fails under linux : if no primary,
        // settle for the first one in the list
        if (! g_primary) {
            int count=0;
            GLFWmonitor ** monitors = glfwGetMonitors(&count);

            if (count)
                g_primary = monitors[0];
        }

        if (g_primary) {
            GLFWvidmode const * vidmode = glfwGetVideoMode(g_primary);
            g_width = vidmode->width;
            g_height = vidmode->height;
        }
    }

    if (! (g_window=glfwCreateWindow(g_width, g_height, windowTitle,
                                       fullscreen && g_primary ? g_primary : NULL, NULL))) {
        printf("Failed to open window.\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(g_window);

    // accommodate high DPI displays (e.g. mac retina displays)
    glfwGetFramebufferSize(g_window, &g_width, &g_height);
    glfwSetFramebufferSizeCallback(g_window, reshape);

    glfwSetKeyCallback(g_window, keyboard);
    glfwSetCursorPosCallback(g_window, motion);
    glfwSetMouseButtonCallback(g_window, mouse);
    glfwSetWindowCloseCallback(g_window, windowClose);

    GLUtils::InitializeGL();
    GLUtils::PrintGLVersion();

    initGL();

    glfwSwapInterval(0);

    initHUD();
    rebuildOsdMeshes();

    GLUtils::CheckGLErrors("before loop");
    while (g_running) {
        idle();
        display();

        glfwPollEvents();
        glfwSwapBuffers(g_window);

        glFinish();
    }

    uninitGL();
    glfwTerminate();
}

//------------------------------------------------------------------------------
