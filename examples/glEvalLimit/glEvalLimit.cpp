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

#include <opensubdiv/osd/cpuEvaluator.h>
#include <opensubdiv/osd/cpuVertexBuffer.h>
#include <opensubdiv/osd/cpuPatchTable.h>
#include <opensubdiv/osd/cpuGLVertexBuffer.h>
#include <opensubdiv/osd/mesh.h>

#ifdef OPENSUBDIV_HAS_TBB
    #include <opensubdiv/osd/tbbEvaluator.h>
#endif

#ifdef OPENSUBDIV_HAS_OPENMP
    #include <opensubdiv/osd/ompEvaluator.h>
#endif

#ifdef OPENSUBDIV_HAS_CUDA
    #include <opensubdiv/osd/cudaEvaluator.h>
    #include <opensubdiv/osd/cudaVertexBuffer.h>
    #include <opensubdiv/osd/cudaGLVertexBuffer.h>
    #include <opensubdiv/osd/cudaPatchTable.h>
    #include "../common/cudaDeviceContext.h"

    CudaDeviceContext g_cudaDeviceContext;
#endif

#ifdef OPENSUBDIV_HAS_OPENCL
    #include <opensubdiv/osd/clVertexBuffer.h>
    #include <opensubdiv/osd/clGLVertexBuffer.h>
    #include <opensubdiv/osd/clEvaluator.h>
    #include <opensubdiv/osd/clPatchTable.h>
    #include "../common/clDeviceContext.h"
    CLDeviceContext g_clDeviceContext;
#endif

#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    #include <opensubdiv/osd/glXFBEvaluator.h>
    #include <opensubdiv/osd/glVertexBuffer.h>
    #include <opensubdiv/osd/glPatchTable.h>
#endif

#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    #include <opensubdiv/osd/glComputeEvaluator.h>
    #include <opensubdiv/osd/glVertexBuffer.h>
    #include <opensubdiv/osd/glPatchTable.h>
#endif

#include <opensubdiv/far/topologyRefiner.h>
#include <opensubdiv/far/stencilTableFactory.h>
#include <opensubdiv/far/patchTableFactory.h>

#include <opensubdiv/far/error.h>

#include "../../regression/common/far_utils.h"
#include "../../regression/common/arg_utils.h"
#include "../common/viewerArgsUtils.h"
#include "../common/stopwatch.h"
#include "../common/simple_math.h"
#include "../common/glControlMeshDisplay.h"
#include "../common/glHud.h"
#include "../common/glUtils.h"

#include "init_shapes.h"
#include "particles.h"

#include <cfloat>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>

using namespace OpenSubdiv;

//------------------------------------------------------------------------------
enum KernelType { kCPU = 0,
                  kOPENMP = 1,
                  kTBB = 2,
                  kCUDA = 3,
                  kCL = 4,
                  kGLXFB = 5,
                  kGLCompute = 6 };

enum EndCap      { kEndCapBilinearBasis,
                   kEndCapBSplineBasis,
                   kEndCapGregoryBasis };

enum HudCheckBox { kHUD_CB_DISPLAY_CONTROL_MESH_EDGES,
                   kHUD_CB_DISPLAY_CONTROL_MESH_VERTS,
                   kHUD_CB_ANIMATE_VERTICES,
                   kHUD_CB_ANIMATE_PARTICLES,
                   kHUD_CB_RANDOM_START,
                   kHUD_CB_FREEZE,
                   kHUD_CB_ADAPTIVE,
                   kHUD_CB_SMOOTH_CORNER_PATCH,
                   kHUD_CB_SINGLE_CREASE_PATCH,
                   kHUD_CB_INF_SHARP_PATCH };

enum DrawMode { kUV,
                kVARYING,
                kNORMAL,
                kSHADE,
                kFACEVARYING,
                kMEAN_CURVATURE };

std::vector<float> g_orgPositions,
                   g_positions,
                   g_varyingColors;

int g_currentShape = 0,
    g_adaptive = 1,
    g_level = 2,
    g_kernel = kCPU,
    g_endCap = kEndCapGregoryBasis,
    g_smoothCornerPatch = 1,
    g_singleCreasePatch = 0,
    g_infSharpPatch = 1,
    g_numElements = 3;

int   g_running = 1,
      g_width = 1024,
      g_height = 1024,
      g_fullscreen = 0,
      g_drawMode = kUV,
      g_prev_x = 0,
      g_prev_y = 0,
      g_mbutton[3] = {0, 0, 0},
      g_frame=0,
      g_freeze=0,
      g_repeatCount;

float g_rotate[2] = {0, 0},
      g_dolly = 5,
      g_pan[2] = {0, 0},
      g_center[3] = {0, 0, 0},
      g_size = 0,
      g_moveScale = 0.0f;

bool  g_yup = false;

GLuint g_transformUB = 0,
       g_transformBinding = 0;

struct Transform {
    float ModelViewMatrix[16];
    float ProjectionMatrix[16];
    float ModelViewProjectionMatrix[16];
} g_transformData;

OpenSubdiv::Sdc::Options::FVarLinearInterpolation  g_fvarBoundary =
    //OpenSubdiv::Sdc::Options::FVAR_LINEAR_ALL;
    OpenSubdiv::Sdc::Options::FVAR_LINEAR_CORNERS_ONLY;

// performance
float g_evalTime = 0;
float g_computeTime = 0;
float g_prevTime = 0;
float g_currentTime = 0;
Stopwatch g_fpsTimer;

//------------------------------------------------------------------------------
int g_nParticles = 65536;

bool g_randomStart = true;//false;
bool g_animParticles = true;

GLuint g_samplesVAO=0;

GLhud g_hud;
GLControlMeshDisplay g_controlMeshDisplay;

//------------------------------------------------------------------------------
struct Program {
    GLuint program;
    GLuint uniformModelViewMatrix;
    GLuint uniformProjectionMatrix;
    GLuint uniformDrawMode;
    GLuint attrPosition;
    GLuint attrColor;
    GLuint attrDu;
    GLuint attrDv;
    GLuint attrDuu;
    GLuint attrDuv;
    GLuint attrDvv;
    GLuint attrPatchCoord;
    GLuint attrFVarData;
} g_defaultProgram;

//------------------------------------------------------------------------------
static void
createRandomColors(int nverts, int stride, float * colors) {

    // large Pell prime number
    srand( static_cast<int>(2147483647) );

    for (int i=0; i<nverts; ++i) {
        colors[i*stride+0] = (float)rand()/(float)RAND_MAX;
        colors[i*stride+1] = (float)rand()/(float)RAND_MAX;
        colors[i*stride+2] = (float)rand()/(float)RAND_MAX;
    }
}

//------------------------------------------------------------------------------
Far::PatchTable const * g_patchTable = NULL;

// input and output vertex data
class EvalOutputBase {
public:
    virtual ~EvalOutputBase() {}
    virtual GLuint BindSourceData() const = 0;
    virtual GLuint BindVertexData() const = 0;
    virtual GLuint Bind1stDerivatives() const = 0;
    virtual GLuint Bind2ndDerivatives() const = 0;
    virtual GLuint BindFaceVaryingData() const = 0;
    virtual GLuint BindPatchCoords() const = 0;
    virtual void UpdateData(const float *src, int startVertex, int numVertices) = 0;
    virtual void UpdateVaryingData(const float *src, int startVertex, int numVertices) = 0;
    virtual void UpdateFaceVaryingData(const float *src, int startVertex, int numVertices) = 0;
    virtual bool HasFaceVaryingData() const = 0;
    virtual void Refine() = 0;
    virtual void EvalPatches() = 0;
    virtual void EvalPatchesWith1stDerivatives() = 0;
    virtual void EvalPatchesWith2ndDerivatives() = 0;
    virtual void EvalPatchesVarying() = 0;
    virtual void EvalPatchesFaceVarying() = 0;
    virtual void UpdatePatchCoords(
        std::vector<Osd::PatchCoord> const &patchCoords) = 0;
};

// note: Since we don't have a class for device-patchcoord container in osd,
// we cheat to use vertexbuffer as a patch-coord (5int) container.
//
// Please don't follow the pattern in your actual application.
//
template<typename SRC_VERTEX_BUFFER, typename EVAL_VERTEX_BUFFER,
         typename STENCIL_TABLE, typename PATCH_TABLE, typename EVALUATOR,
         typename DEVICE_CONTEXT = void>
class EvalOutput : public EvalOutputBase {
public:
    typedef OpenSubdiv::Osd::EvaluatorCacheT<EVALUATOR> EvaluatorCache;

    EvalOutput(Far::StencilTable const *vertexStencils,
               Far::StencilTable const *varyingStencils,
               Far::StencilTable const *faceVaryingStencils,
               int fvarChannel, int fvarWidth,
               int numParticles, Far::PatchTable const *patchTable,
               EvaluatorCache *evaluatorCache = NULL,
               DEVICE_CONTEXT *deviceContext = NULL)
        : _srcDesc(       /*offset*/ 0, /*length*/ 3, /*stride*/ 3),
          _srcVaryingDesc(/*offset*/ 0, /*length*/ 3, /*stride*/ 3),
          _srcFVarDesc(   /*offset*/ 0, /*length*/ fvarWidth, /*stride*/ fvarWidth),
          _vertexDesc(    /*offset*/ 0, /*length*/ 3, /*stride*/ 6),
          _varyingDesc(   /*offset*/ 3, /*length*/ 3, /*stride*/ 6),
          _fvarDesc(      /*offset*/ 0, /*length*/ fvarWidth, /*stride*/ fvarWidth),
          _duDesc(        /*offset*/ 0, /*length*/ 3, /*stride*/ 6),
          _dvDesc(        /*offset*/ 3, /*length*/ 3, /*stride*/ 6),
          _duuDesc(       /*offset*/ 0, /*length*/ 3, /*stride*/ 9),
          _duvDesc(       /*offset*/ 3, /*length*/ 3, /*stride*/ 9),
          _dvvDesc(       /*offset*/ 6, /*length*/ 3, /*stride*/ 9),
          _deviceContext(deviceContext) {

        // total number of vertices = coarse points + refined points + local points
        int numTotalVerts = vertexStencils->GetNumControlVertices()
                          + vertexStencils->GetNumStencils();

        _srcData = SRC_VERTEX_BUFFER::Create(3, numTotalVerts, _deviceContext);
        _srcVaryingData = SRC_VERTEX_BUFFER::Create(3, numTotalVerts, _deviceContext);
        _vertexData = EVAL_VERTEX_BUFFER::Create(6, numParticles, _deviceContext);
        _deriv1 = EVAL_VERTEX_BUFFER::Create(6, numParticles, _deviceContext);
        _deriv2 = EVAL_VERTEX_BUFFER::Create(9, numParticles, _deviceContext);
        _patchTable = PATCH_TABLE::Create(patchTable, _deviceContext);
        _patchCoords = NULL;
        _numCoarseVerts = vertexStencils->GetNumControlVertices();
        _vertexStencils =
            Osd::convertToCompatibleStencilTable<STENCIL_TABLE>(vertexStencils, _deviceContext);
        _varyingStencils =
            Osd::convertToCompatibleStencilTable<STENCIL_TABLE>(varyingStencils, _deviceContext);

        if (faceVaryingStencils) {
            _numCoarseFVarVerts = faceVaryingStencils->GetNumControlVertices();
            int numTotalFVarVerts = faceVaryingStencils->GetNumControlVertices()
                                  + faceVaryingStencils->GetNumStencils();
            _srcFVarData = EVAL_VERTEX_BUFFER::Create(2, numTotalFVarVerts, _deviceContext);
            _fvarData = EVAL_VERTEX_BUFFER::Create(fvarWidth, numParticles, _deviceContext);
            _faceVaryingStencils =
                Osd::convertToCompatibleStencilTable<STENCIL_TABLE>(faceVaryingStencils, _deviceContext);
            _fvarChannel = fvarChannel;
            _fvarWidth = fvarWidth;
        } else {
            _numCoarseFVarVerts = 0;
            _srcFVarData = NULL;
            _fvarData = NULL;
            _faceVaryingStencils = NULL;
            _fvarChannel = 0;
            _fvarWidth = 0;
        }
        _evaluatorCache = evaluatorCache;
    }
    ~EvalOutput() {
        delete _srcData;
        delete _srcVaryingData;
        delete _srcFVarData;
        delete _vertexData;
        delete _deriv1;
        delete _deriv2;
        delete _fvarData;
        delete _patchTable;
        delete _patchCoords;
        delete _vertexStencils;
        delete _varyingStencils;
        delete _faceVaryingStencils;
    }
    virtual GLuint BindSourceData() const {
        return _srcData->BindVBO();
    }
    virtual GLuint BindVertexData() const {
        return _vertexData->BindVBO();
    }
    virtual GLuint Bind1stDerivatives() const {
        return _deriv1->BindVBO();
    }
    virtual GLuint Bind2ndDerivatives() const {
        return _deriv2->BindVBO();
    }
    virtual GLuint BindFaceVaryingData() const {
        return _fvarData->BindVBO();
    }
    virtual GLuint BindPatchCoords() const {
        return _patchCoords->BindVBO();
    }
    virtual void UpdateData(const float *src, int startVertex, int numVertices) {
        _srcData->UpdateData(src, startVertex, numVertices, _deviceContext);
    }
    virtual void UpdateVaryingData(const float *src, int startVertex, int numVertices) {
        _srcVaryingData->UpdateData(src, startVertex, numVertices, _deviceContext);
    }
    virtual void UpdateFaceVaryingData(const float *src, int startVertex, int numVertices) {
        _srcFVarData->UpdateData(src, startVertex, numVertices, _deviceContext);
    }
    virtual bool HasFaceVaryingData() const {
        return _faceVaryingStencils != NULL;
    }
    virtual void Refine() {
        Osd::BufferDescriptor dstDesc = _srcDesc;
        dstDesc.offset += _numCoarseVerts * _srcDesc.stride;

        EVALUATOR const *evalInstance = OpenSubdiv::Osd::GetEvaluator<EVALUATOR>(
            _evaluatorCache, _srcDesc, dstDesc, _deviceContext);

        EVALUATOR::EvalStencils(_srcData, _srcDesc,
                                _srcData, dstDesc,
                                _vertexStencils,
                                evalInstance,
                                _deviceContext);

        dstDesc = _srcVaryingDesc;
        dstDesc.offset += _numCoarseVerts * _srcVaryingDesc.stride;
        evalInstance = OpenSubdiv::Osd::GetEvaluator<EVALUATOR>(
            _evaluatorCache, _srcVaryingDesc, dstDesc, _deviceContext);

        EVALUATOR::EvalStencils(_srcVaryingData, _srcVaryingDesc,
                                _srcVaryingData, dstDesc,
                                _varyingStencils,
                                evalInstance,
                                _deviceContext);

        if (HasFaceVaryingData()) {
            Osd::BufferDescriptor dstFVarDesc = _srcFVarDesc;
            dstFVarDesc.offset += _numCoarseFVarVerts * _srcFVarDesc.stride;

            evalInstance = OpenSubdiv::Osd::GetEvaluator<EVALUATOR>(
                _evaluatorCache, _srcFVarDesc, dstFVarDesc, _deviceContext);

            EVALUATOR::EvalStencils(_srcFVarData, _srcFVarDesc,
                                    _srcFVarData, dstFVarDesc,
                                    _faceVaryingStencils,
                                    evalInstance,
                                    _deviceContext);
        }

    }
    virtual void EvalPatches() {
        EVALUATOR const *evalInstance = OpenSubdiv::Osd::GetEvaluator<EVALUATOR>(
            _evaluatorCache, _srcDesc, _vertexDesc, _deviceContext);

        EVALUATOR::EvalPatches(
            _srcData, _srcDesc,
            _vertexData, _vertexDesc,
            _patchCoords->GetNumVertices(),
            _patchCoords,
            _patchTable, evalInstance, _deviceContext);
    }
    virtual void EvalPatchesWith1stDerivatives() {
        EVALUATOR const *evalInstance = OpenSubdiv::Osd::GetEvaluator<EVALUATOR>(
            _evaluatorCache, _srcDesc, _vertexDesc, _duDesc, _dvDesc, _deviceContext);
        EVALUATOR::EvalPatches(
            _srcData, _srcDesc,
            _vertexData, _vertexDesc,
            _deriv1, _duDesc,
            _deriv1, _dvDesc,
            _patchCoords->GetNumVertices(),
            _patchCoords,
            _patchTable, evalInstance, _deviceContext);
    }
    virtual void EvalPatchesWith2ndDerivatives() {
        EVALUATOR const *evalInstance = OpenSubdiv::Osd::GetEvaluator<EVALUATOR>(
            _evaluatorCache, _srcDesc, _vertexDesc,
            _duDesc, _dvDesc, _duuDesc, _duvDesc, _dvvDesc,
            _deviceContext);
        EVALUATOR::EvalPatches(
            _srcData, _srcDesc,
            _vertexData, _vertexDesc,
            _deriv1, _duDesc,
            _deriv1, _dvDesc,
            _deriv2, _duuDesc,
            _deriv2, _duvDesc,
            _deriv2, _dvvDesc,
            _patchCoords->GetNumVertices(),
            _patchCoords,
            _patchTable, evalInstance, _deviceContext);
    }
    virtual void EvalPatchesVarying() {
        EVALUATOR const *evalInstance = OpenSubdiv::Osd::GetEvaluator<EVALUATOR>(
            _evaluatorCache, _srcVaryingDesc, _varyingDesc, _deviceContext);

        EVALUATOR::EvalPatchesVarying(
            _srcVaryingData, _srcVaryingDesc,
            // varying data is interleaved in vertexData.
            _vertexData, _varyingDesc,
            _patchCoords->GetNumVertices(),
            _patchCoords,
            _patchTable, evalInstance, _deviceContext);
    }
    virtual void EvalPatchesFaceVarying() {
        EVALUATOR const *evalInstance = OpenSubdiv::Osd::GetEvaluator<EVALUATOR>(
            _evaluatorCache, _srcFVarDesc, _fvarDesc, _deviceContext);

        EVALUATOR::EvalPatchesFaceVarying(
            _srcFVarData, _srcFVarDesc,
            _fvarData, _fvarDesc,
            _patchCoords->GetNumVertices(),
            _patchCoords,
            _patchTable, _fvarChannel, evalInstance, _deviceContext);
    }
    virtual void UpdatePatchCoords(
        std::vector<Osd::PatchCoord> const &patchCoords) {
        if (_patchCoords &&
            _patchCoords->GetNumVertices() != (int)patchCoords.size()) {
            delete _patchCoords;
            _patchCoords = NULL;
        }
        if (! _patchCoords) {
            _patchCoords = EVAL_VERTEX_BUFFER::Create(5,
                                                      (int)patchCoords.size(),
                                                      _deviceContext);
        }
        _patchCoords->UpdateData((float*)&patchCoords[0], 0, (int)patchCoords.size(), _deviceContext);
    }
private:
    SRC_VERTEX_BUFFER *_srcData;
    SRC_VERTEX_BUFFER *_srcVaryingData;
    EVAL_VERTEX_BUFFER *_srcFVarData;
    EVAL_VERTEX_BUFFER *_vertexData;
    EVAL_VERTEX_BUFFER *_deriv1;
    EVAL_VERTEX_BUFFER *_deriv2;
    EVAL_VERTEX_BUFFER *_fvarData;
    EVAL_VERTEX_BUFFER *_patchCoords;
    PATCH_TABLE *_patchTable;
    Osd::BufferDescriptor _srcDesc;
    Osd::BufferDescriptor _srcVaryingDesc;
    Osd::BufferDescriptor _srcFVarDesc;
    Osd::BufferDescriptor _vertexDesc;
    Osd::BufferDescriptor _varyingDesc;
    Osd::BufferDescriptor _fvarDesc;
    Osd::BufferDescriptor _duDesc;
    Osd::BufferDescriptor _dvDesc;
    Osd::BufferDescriptor _duuDesc;
    Osd::BufferDescriptor _duvDesc;
    Osd::BufferDescriptor _dvvDesc;
    int _numCoarseVerts;
    int _numCoarseFVarVerts;

    STENCIL_TABLE const *_vertexStencils;
    STENCIL_TABLE const *_varyingStencils;
    STENCIL_TABLE const *_faceVaryingStencils;

    int _fvarChannel;
    int _fvarWidth;

    EvaluatorCache *_evaluatorCache;
    DEVICE_CONTEXT *_deviceContext;
};

// This example uses one shared interleaved buffer for evaluated
// 1st derivatives and a second shared interleaved buffer for
// evaluated 2nd derivatives. We use this specialized device
// context to allow the XFB evaluator to take advantage of this
// and make more efficient use of available XFB buffer bindings.
struct XFBDeviceContext {
    bool AreInterleavedDerivativeBuffers() const { return true; }
} g_xfbDeviceContext;

EvalOutputBase *g_evalOutput = NULL;
STParticles * g_particles=0;

//------------------------------------------------------------------------------
static void
updateGeom() {
    int nverts = (int)g_orgPositions.size() / 3;

    const float *p = &g_orgPositions[0];

    float r = sin(g_frame*0.1f) * g_moveScale;

    for (int i = 0; i < nverts; ++i) {
        //float move = 0.05f*cosf(p[0]*20+g_frame*0.01f);
        float ct = cos(p[2] * r);
        float st = sin(p[2] * r);
        g_positions[i*3+0] = p[0]*ct + p[1]*st;
        g_positions[i*3+1] = -p[0]*st + p[1]*ct;
        g_positions[i*3+2] = p[2];
        p+=3;
    }

    // Run Compute pass to pose the control vertices ---------------------------
    Stopwatch s;
    s.Start();

    // update coarse vertices
    g_evalOutput->UpdateData(&g_positions[0], 0, nverts);

    // update coarse varying
    if (g_drawMode == kVARYING) {
        g_evalOutput->UpdateVaryingData(&g_varyingColors[0], 0, nverts);

    }

    // Refine
    g_evalOutput->Refine();

    s.Stop();
    g_computeTime = float(s.GetElapsed() * 1000.0f);


    // Run Eval pass to get the samples locations ------------------------------

    s.Start();

    // Apply 'dynamics' update
    assert(g_particles);

    float elapsed = g_currentTime - g_prevTime;
    g_particles->Update(elapsed);
    g_prevTime = g_currentTime;

    std::vector<OpenSubdiv::Osd::PatchCoord> const &patchCoords
        = g_particles->GetPatchCoords();

    // update patchcoord to be evaluated
    g_evalOutput->UpdatePatchCoords(patchCoords);

    // Evaluate the positions of the samples on the limit surface
    if (g_drawMode == kMEAN_CURVATURE) {
        // evaluate positions and 2nd derivatives
        g_evalOutput->EvalPatchesWith2ndDerivatives();
    } else if (g_drawMode == kNORMAL || g_drawMode == kSHADE) {
        // evaluate positions and 1st derivatives
        g_evalOutput->EvalPatchesWith1stDerivatives();
    } else {
        // evaluate positions
        g_evalOutput->EvalPatches();
    }

    // color
    if (g_drawMode == kVARYING) {
        g_evalOutput->EvalPatchesVarying();
    } else if (g_drawMode == kFACEVARYING && g_evalOutput->HasFaceVaryingData()) {
        g_evalOutput->EvalPatchesFaceVarying();
    }

    s.Stop();

    g_evalTime = float(s.GetElapsed());
}

//------------------------------------------------------------------------------
static void
createOsdMesh(ShapeDesc const & shapeDesc, int level) {

    Shape * shape = Shape::parseObj(shapeDesc);

    // create Far mesh (topology)
    Sdc::SchemeType sdctype = GetSdcType(*shape);
    Sdc::Options sdcoptions = GetSdcOptions(*shape);

    sdcoptions.SetFVarLinearInterpolation(g_fvarBoundary);

    Far::TopologyRefiner *topologyRefiner =
        Far::TopologyRefinerFactory<Shape>::Create(*shape,
            Far::TopologyRefinerFactory<Shape>::Options(sdctype, sdcoptions));

    g_orgPositions=shape->verts;
    g_positions.resize(g_orgPositions.size(), 0.0f);

    // compute model bounding
    float min[3] = { FLT_MAX,  FLT_MAX,  FLT_MAX};
    float max[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (size_t i=0; i <g_orgPositions.size()/3; ++i) {
        for(int j=0; j<3; ++j) {
            float v = g_orgPositions[i*3+j];
            min[j] = std::min(min[j], v);
            max[j] = std::max(max[j], v);
        }
    }
    for (int j=0; j<3; ++j) {
        g_center[j] = (min[j] + max[j]) * 0.5f;
        g_size += (max[j]-min[j])*(max[j]-min[j]);
    }
    g_size = sqrtf(g_size);


    float speed = g_particles ? g_particles->GetSpeed() : 0.2f;

    // save coarse topology (used for coarse mesh drawing)
    g_controlMeshDisplay.SetTopology(topologyRefiner->GetLevel(0));

    // create random varying color
    {
        int numCoarseVerts = topologyRefiner->GetLevel(0).GetNumVertices();
        g_varyingColors.resize(numCoarseVerts*3);
        createRandomColors(numCoarseVerts, 3, &g_varyingColors[0]);
    }

    Far::StencilTable const * vertexStencils = NULL;
    Far::StencilTable const * varyingStencils = NULL;
    Far::StencilTable const * faceVaryingStencils = NULL;

    int fvarChannel = 0;
    int fvarWidth = shape->GetFVarWidth();
    bool hasFVarData = !shape->uvs.empty();

    {
        if (g_adaptive) {
            // Apply feature adaptive refinement to the mesh so that we can use the
            // limit evaluation API features.
            Far::TopologyRefiner::AdaptiveOptions options(level);
            options.considerFVarChannels = hasFVarData;
            options.useSingleCreasePatch = g_singleCreasePatch;
            options.useInfSharpPatch = g_infSharpPatch;
            topologyRefiner->RefineAdaptive(options);
        } else {
            Far::TopologyRefiner::UniformOptions options(level);
            topologyRefiner->RefineUniform(options);
        }

        // Generate stencil table to update the bi-cubic patches control
        // vertices after they have been re-posed (both for vertex & varying
        // interpolation)
        Far::StencilTableFactory::Options soptions;
        soptions.generateOffsets=true;
        soptions.generateIntermediateLevels=g_adaptive;

        vertexStencils =
            Far::StencilTableFactory::Create(*topologyRefiner, soptions);

        soptions.interpolationMode = Far::StencilTableFactory::INTERPOLATE_VARYING;
        varyingStencils =
            Far::StencilTableFactory::Create(*topologyRefiner, soptions);

        if (hasFVarData) {
            soptions.interpolationMode = Far::StencilTableFactory::INTERPOLATE_FACE_VARYING;
            soptions.fvarChannel = fvarChannel;
            faceVaryingStencils =
                Far::StencilTableFactory::Create(*topologyRefiner, soptions);
        }

        // Generate bi-cubic patch table for the limit surface
        Far::PatchTableFactory::Options poptions(level);
        if (g_endCap == kEndCapBilinearBasis) {
            poptions.SetEndCapType(
                Far::PatchTableFactory::Options::ENDCAP_BILINEAR_BASIS);
        } else if (g_endCap == kEndCapBSplineBasis) {
            poptions.SetEndCapType(
                Far::PatchTableFactory::Options::ENDCAP_BSPLINE_BASIS);
        } else {
            poptions.SetEndCapType(
                Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS);
        }
        poptions.generateLegacySharpCornerPatches = !g_smoothCornerPatch;
        poptions.useSingleCreasePatch = g_singleCreasePatch;
        poptions.useInfSharpPatch = g_infSharpPatch;
        poptions.generateFVarTables = hasFVarData;
        poptions.generateFVarLegacyLinearPatches = false;
        poptions.includeFVarBaseLevelIndices = true;;

        Far::PatchTable const * patchTable =
            Far::PatchTableFactory::Create(*topologyRefiner, poptions);

        // append local points stencils
        if (Far::StencilTable const *localPointStencilTable =
            patchTable->GetLocalPointStencilTable()) {
            Far::StencilTable const *table =
                Far::StencilTableFactory::AppendLocalPointStencilTable(
                    *topologyRefiner, vertexStencils, localPointStencilTable);
            delete vertexStencils;
            vertexStencils = table;
        }
        if (Far::StencilTable const *localPointVaryingStencilTable =
            patchTable->GetLocalPointVaryingStencilTable()) {
            Far::StencilTable const *table =
                Far::StencilTableFactory::AppendLocalPointStencilTable(
                    *topologyRefiner,
                    varyingStencils, localPointVaryingStencilTable);
            delete varyingStencils;
            varyingStencils = table;
        }
        if (Far::StencilTable const *localPointFaceVaryingStencilTable =
            patchTable->GetLocalPointFaceVaryingStencilTable()) {
            Far::StencilTable const *table =
                Far::StencilTableFactory::AppendLocalPointStencilTableFaceVarying(
                    *topologyRefiner,
                    faceVaryingStencils, localPointFaceVaryingStencilTable);
            delete faceVaryingStencils;
            faceVaryingStencils = table;
        }

        if (g_patchTable) delete g_patchTable;
        g_patchTable = patchTable;
    }

    // In following template instantiations, same type of vertex buffers are
    // used for both source and destination (first and second template
    // parameters), since we'd like to draw control mesh wireframe too in
    // this example viewer.
    // If we don't need to draw the coarse control mesh, the src buffer doesn't
    // have to be interoperable to GL (it can be CpuVertexBuffer etc).

    delete g_evalOutput;
    if (g_kernel == kCPU) {
        g_evalOutput = new EvalOutput<Osd::CpuGLVertexBuffer,
                                      Osd::CpuGLVertexBuffer,
                                      Far::StencilTable,
                                      Osd::CpuPatchTable,
                                      Osd::CpuEvaluator>
            (vertexStencils, varyingStencils, faceVaryingStencils,
             fvarChannel, fvarWidth,
             g_nParticles, g_patchTable);
#ifdef OPENSUBDIV_HAS_OPENMP
    } else if (g_kernel == kOPENMP) {
        g_evalOutput = new EvalOutput<Osd::CpuGLVertexBuffer,
                                      Osd::CpuGLVertexBuffer,
                                      Far::StencilTable,
                                      Osd::CpuPatchTable,
                                      Osd::OmpEvaluator>
            (vertexStencils, varyingStencils, faceVaryingStencils,
            fvarChannel, fvarWidth,
            g_nParticles, g_patchTable);
#endif
#ifdef OPENSUBDIV_HAS_TBB
    } else if (g_kernel == kTBB) {
        g_evalOutput = new EvalOutput<Osd::CpuGLVertexBuffer,
                                      Osd::CpuGLVertexBuffer,
                                      Far::StencilTable,
                                      Osd::CpuPatchTable,
                                      Osd::TbbEvaluator>
            (vertexStencils, varyingStencils, faceVaryingStencils,
            fvarChannel, fvarWidth,
            g_nParticles, g_patchTable);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    } else if (g_kernel == kCUDA) {
        g_evalOutput = new EvalOutput<Osd::CudaGLVertexBuffer,
                                      Osd::CudaGLVertexBuffer,
                                      Osd::CudaStencilTable,
                                      Osd::CudaPatchTable,
                                      Osd::CudaEvaluator>
            (vertexStencils, varyingStencils, faceVaryingStencils,
            fvarChannel, fvarWidth,
            g_nParticles, g_patchTable);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    } else if (g_kernel == kCL) {
        static Osd::EvaluatorCacheT<Osd::CLEvaluator> clEvaluatorCache;
        g_evalOutput = new EvalOutput<Osd::CLGLVertexBuffer,
                                      Osd::CLGLVertexBuffer,
                                      Osd::CLStencilTable,
                                      Osd::CLPatchTable,
                                      Osd::CLEvaluator,
                                      CLDeviceContext>
            (vertexStencils, varyingStencils, faceVaryingStencils,
            fvarChannel, fvarWidth,
            g_nParticles, g_patchTable,
            &clEvaluatorCache, &g_clDeviceContext);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    } else if (g_kernel == kGLXFB) {
        static Osd::EvaluatorCacheT<Osd::GLXFBEvaluator> glXFBEvaluatorCache;
        g_evalOutput = new EvalOutput<Osd::GLVertexBuffer,
                                      Osd::GLVertexBuffer,
                                      Osd::GLStencilTableTBO,
                                      Osd::GLPatchTable,
                                      Osd::GLXFBEvaluator,
                                      XFBDeviceContext>
            (vertexStencils, varyingStencils, faceVaryingStencils,
            fvarChannel, fvarWidth,
            g_nParticles, g_patchTable,
             &glXFBEvaluatorCache, &g_xfbDeviceContext);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    } else if (g_kernel == kGLCompute) {
        static Osd::EvaluatorCacheT<Osd::GLComputeEvaluator> glComputeEvaluatorCache;
        g_evalOutput = new EvalOutput<Osd::GLVertexBuffer,
                                      Osd::GLVertexBuffer,
                                      Osd::GLStencilTableSSBO,
                                      Osd::GLPatchTable,
                                      Osd::GLComputeEvaluator>
            (vertexStencils, varyingStencils, faceVaryingStencils,
            fvarChannel, fvarWidth,
            g_nParticles, g_patchTable,
             &glComputeEvaluatorCache);
#endif
    }

    if (g_evalOutput->HasFaceVaryingData()) {
        g_evalOutput->UpdateFaceVaryingData(
            &shape->uvs[0], 0, (int)shape->uvs.size()/shape->GetFVarWidth());
    }

    delete shape;

    // Create the 'uv particles' manager - this class manages the limit
    // location samples (ptex face index, (s,t) and updates them between frames.
    // Note: the number of limit locations can be entirely arbitrary
    delete g_particles;
    g_particles = new STParticles(*topologyRefiner, g_patchTable,
                                  g_nParticles, !g_randomStart);
    g_nParticles = g_particles->GetNumParticles();
    g_particles->SetSpeed(speed);

    g_prevTime = -1;
    g_currentTime = 0;

    updateGeom();

    delete topologyRefiner;
}

//------------------------------------------------------------------------------
static bool
linkDefaultProgram() {

#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
    #define GLSL_VERSION_DEFINE "#version 400\n"
#else
    #define GLSL_VERSION_DEFINE "#version 150\n"
#endif

    static const char *vsSrc =
        GLSL_VERSION_DEFINE
        "in vec3 position;\n"
        "in vec3 color;\n"
        "in vec3 du;\n"
        "in vec3 dv;\n"
        "in vec3 duu;\n"
        "in vec3 duv;\n"
        "in vec3 dvv;\n"
        "in vec2 patchCoord;\n"
        "in vec2 fvarData;\n"
        "out vec4 fragColor;\n"
        "uniform mat4 ModelViewMatrix;\n"
        "uniform mat4 ProjectionMatrix;\n"
        "uniform int DrawMode;\n"
        "void main() {\n"
        "  vec3 normal = (ModelViewMatrix * "
        "               vec4(normalize(cross(du, dv)), 0)).xyz;\n"
        "  gl_Position = ProjectionMatrix * ModelViewMatrix * "
        "                  vec4(position, 1);\n"
        "  if (DrawMode == 0) {\n" // UV
        "    fragColor = vec4(patchCoord.x, patchCoord.y, 0, 1);\n"
        "  } else if (DrawMode == 2) {\n"
        "    fragColor = vec4(normal*0.5+vec3(0.5), 1);\n"
        "  } else if (DrawMode == 3) {\n"
        "    fragColor = vec4(vec3(1)*dot(normal, vec3(0,0,1)), 1);\n"
        "  } else if (DrawMode == 4) {\n"  // face varying
        "    // generating a checkerboard pattern\n"
        "    int checker = int(floor(20*fvarData.r)+floor(20*fvarData.g))&1;\n"
        "    fragColor = vec4(fvarData.rg*checker, 1-checker, 1);\n"
        "  } else if (DrawMode == 5) {\n"  // mean curvature
        "    vec3 N = normalize(cross(du, dv));\n"
        "    float E = dot(du, du);\n"
        "    float F = dot(du, dv);\n"
        "    float G = dot(dv, dv);\n"
        "    float e = dot(N, duu);\n"
        "    float f = dot(N, duv);\n"
        "    float g = dot(N, dvv);\n"
        "    float H = 0.5 * abs(0.5 * (E*g - 2*F*f - G*e) / (E*G - F*F));\n"
        "    fragColor = vec4(H, H, H, 1.0);\n"
        "  } else {\n" // varying
        "    fragColor = vec4(color, 1);\n"
        "  }\n"
        "}\n";

    static const char *fsSrc =
        GLSL_VERSION_DEFINE
        "in vec4 fragColor;\n"
        "out vec4 color;\n"
        "void main() {\n"
        "  color = fragColor;\n"
        "}\n";

    GLuint program = glCreateProgram();
    GLuint vertexShader = GLUtils::CompileShader(GL_VERTEX_SHADER, vsSrc);
    GLuint fragmentShader = GLUtils::CompileShader(GL_FRAGMENT_SHADER, fsSrc);

    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);

    glBindAttribLocation(program, 0, "position");
    glBindAttribLocation(program, 1, "color");
    glBindAttribLocation(program, 2, "du");
    glBindAttribLocation(program, 3, "dv");
    glBindAttribLocation(program, 4, "duu");
    glBindAttribLocation(program, 5, "duv");
    glBindAttribLocation(program, 6, "dvv");
    glBindAttribLocation(program, 7, "patchCoord");
    glBindAttribLocation(program, 8, "fvarData");
    glBindFragDataLocation(program, 0, "color");

    glLinkProgram(program);

    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status == GL_FALSE) {
        GLint infoLogLength;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);
        char *infoLog = new char[infoLogLength];
        glGetProgramInfoLog(program, infoLogLength, NULL, infoLog);
        printf("%s\n", infoLog);
        delete[] infoLog;
        exit(1);
    }

    g_defaultProgram.program = program;
    g_defaultProgram.uniformModelViewMatrix =
        glGetUniformLocation(program, "ModelViewMatrix");
    g_defaultProgram.uniformProjectionMatrix =
        glGetUniformLocation(program, "ProjectionMatrix");
    g_defaultProgram.uniformDrawMode =
        glGetUniformLocation(program, "DrawMode");
    g_defaultProgram.attrPosition = glGetAttribLocation(program, "position");
    g_defaultProgram.attrColor = glGetAttribLocation(program, "color");
    g_defaultProgram.attrDu = glGetAttribLocation(program, "du");
    g_defaultProgram.attrDv = glGetAttribLocation(program, "dv");
    g_defaultProgram.attrDuu = glGetAttribLocation(program, "duu");
    g_defaultProgram.attrDuv = glGetAttribLocation(program, "duv");
    g_defaultProgram.attrDvv = glGetAttribLocation(program, "dvv");
    g_defaultProgram.attrPatchCoord = glGetAttribLocation(program, "patchCoord");
    g_defaultProgram.attrFVarData = glGetAttribLocation(program, "fvarData");

    return true;
}

//------------------------------------------------------------------------------
static void
drawSamples() {
    glUseProgram(g_defaultProgram.program);

    glUniformMatrix4fv(g_defaultProgram.uniformModelViewMatrix,
                       1, GL_FALSE, g_transformData.ModelViewMatrix);
    glUniformMatrix4fv(g_defaultProgram.uniformProjectionMatrix,
                       1, GL_FALSE, g_transformData.ProjectionMatrix);
    glUniform1i(g_defaultProgram.uniformDrawMode, g_drawMode);

    glBindVertexArray(g_samplesVAO);

    glEnableVertexAttribArray(g_defaultProgram.attrPosition);
    glEnableVertexAttribArray(g_defaultProgram.attrColor);
    glBindBuffer(GL_ARRAY_BUFFER, g_evalOutput->BindVertexData());
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 6, 0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 6, (float*)12);

    glEnableVertexAttribArray(g_defaultProgram.attrDu);
    glEnableVertexAttribArray(g_defaultProgram.attrDv);
    glBindBuffer(GL_ARRAY_BUFFER, g_evalOutput->Bind1stDerivatives());
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 6, 0);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 6, (float*)12);

    glEnableVertexAttribArray(g_defaultProgram.attrDuu);
    glEnableVertexAttribArray(g_defaultProgram.attrDuv);
    glEnableVertexAttribArray(g_defaultProgram.attrDvv);
    glBindBuffer(GL_ARRAY_BUFFER, g_evalOutput->Bind2ndDerivatives());
    glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 9, 0);
    glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 9, (float*)12);
    glVertexAttribPointer(6, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 9, (float*)24);

    glEnableVertexAttribArray(g_defaultProgram.attrPatchCoord);
    glBindBuffer(GL_ARRAY_BUFFER, g_evalOutput->BindPatchCoords());
    glVertexAttribPointer(7, 2, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 5, (float*)12);

    if (g_evalOutput->HasFaceVaryingData()) {
        glEnableVertexAttribArray(g_defaultProgram.attrFVarData);
        glBindBuffer(GL_ARRAY_BUFFER, g_evalOutput->BindFaceVaryingData());
        glVertexAttribPointer(8, 2, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 2, 0);
    }

    glPointSize(2.0f);
    int nPatchCoords = (int)g_particles->GetPatchCoords().size();
    glDrawArrays(GL_POINTS, 0, nPatchCoords);
    glPointSize(1.0f);

    glDisableVertexAttribArray(g_defaultProgram.attrPosition);
    glDisableVertexAttribArray(g_defaultProgram.attrColor);
    glDisableVertexAttribArray(g_defaultProgram.attrDu);
    glDisableVertexAttribArray(g_defaultProgram.attrDv);
    glDisableVertexAttribArray(g_defaultProgram.attrDuu);
    glDisableVertexAttribArray(g_defaultProgram.attrDuv);
    glDisableVertexAttribArray(g_defaultProgram.attrDvv);
    glDisableVertexAttribArray(g_defaultProgram.attrPatchCoord);
    glDisableVertexAttribArray(g_defaultProgram.attrFVarData);

    glBindVertexArray(0);

    glUseProgram(0);
}

//------------------------------------------------------------------------------
static void
display() {

    Stopwatch s;
    s.Start();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, g_width, g_height);
    g_hud.FillBackground();

    double aspect = g_width/(double)g_height;
    identity(g_transformData.ModelViewMatrix);
    translate(g_transformData.ModelViewMatrix, -g_pan[0], -g_pan[1], -g_dolly);
    rotate(g_transformData.ModelViewMatrix, g_rotate[1], 1, 0, 0);
    rotate(g_transformData.ModelViewMatrix, g_rotate[0], 0, 1, 0);
    if (!g_yup) {
        rotate(g_transformData.ModelViewMatrix, -90, 1, 0, 0);
    }
    translate(g_transformData.ModelViewMatrix,
              -g_center[0], -g_center[1], -g_center[2]);
    perspective(g_transformData.ProjectionMatrix,
                45.0f, (float)aspect, 0.01f, 500.0f);
    multMatrix(g_transformData.ModelViewProjectionMatrix,
               g_transformData.ModelViewMatrix,
               g_transformData.ProjectionMatrix);

    glEnable(GL_DEPTH_TEST);

    s.Stop();
    float drawCpuTime = float(s.GetElapsed() * 1000.0f);
    s.Start();
    glFinish();
    s.Stop();
    float drawGpuTime = float(s.GetElapsed() * 1000.0f);

    drawSamples();

    // draw the control mesh
    g_controlMeshDisplay.Draw(
        g_evalOutput->BindSourceData(), 3*sizeof(float),
        g_transformData.ModelViewProjectionMatrix);

    if (g_hud.IsVisible()) {
        g_fpsTimer.Stop();
        double elapsed = g_fpsTimer.GetElapsed();
        g_fpsTimer.Start();
        double fps = 1.0/elapsed;
        if (g_animParticles) g_currentTime += (float)elapsed;

        int nPatchCoords = (int)g_particles->GetPatchCoords().size();

        g_hud.DrawString(10, -150, "Particle Speed ([) (]): %.1f", g_particles->GetSpeed());
        g_hud.DrawString(10, -120, "# Samples  : (%d / %d)", nPatchCoords, g_nParticles);
        g_hud.DrawString(10, -100, "Compute    : %.3f ms", g_computeTime);
        g_hud.DrawString(10, -80,  "Eval       : %.3f ms", g_evalTime * 1000.f);
        g_hud.DrawString(10, -60,  "GPU Draw   : %.3f ms", drawGpuTime);
        g_hud.DrawString(10, -40,  "CPU Draw   : %.3f ms", drawCpuTime);
        g_hud.DrawString(10, -20,  "FPS        : %3.1f", fps);

        g_hud.Flush();
    }

    glFinish();

    GLUtils::CheckGLErrors("display leave");
}

//------------------------------------------------------------------------------
static void
idle() {

    if (! g_freeze)
        g_frame++;

    updateGeom();

    if (g_repeatCount != 0 && g_frame >= g_repeatCount)
        g_running = 0;
}

//------------------------------------------------------------------------------
static void
motion(GLFWwindow *, double dx, double dy) {

    int x=(int)dx, y=(int)dy;

    if (g_mbutton[0] && !g_mbutton[1] && !g_mbutton[2]) {
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
setSamples(bool add) {
    if (add) {
        g_nParticles = g_nParticles * 2;
    } else {
        g_nParticles = std::max(1, g_nParticles / 2);
    }

    createOsdMesh(g_defaultShapes[g_currentShape], g_level);
}

//------------------------------------------------------------------------------
static void
fitFrame() {

    g_pan[0] = g_pan[1] = 0;
    g_dolly = g_size;
}

//------------------------------------------------------------------------------
static void
keyboard(GLFWwindow *, int key, int /* scancode */, int event, int /* mods */) {

    if (event == GLFW_RELEASE) return;
    if (g_hud.KeyDown(tolower(key))) return;

    switch (key) {
        case 'Q': g_running = 0; break;

        case 'F': fitFrame(); break;

        case '=': setSamples(true); break;

        case '-': setSamples(false); break;

        case '[': if (g_particles) {
                      g_particles->SetSpeed(g_particles->GetSpeed()-0.1f);
                  } break;
        case ']': if (g_particles) {
                      g_particles->SetSpeed(g_particles->GetSpeed()+0.1f);
                  } break;

        case GLFW_KEY_ESCAPE: g_hud.SetVisible(!g_hud.IsVisible()); break;
    }
}

//------------------------------------------------------------------------------
static void
callbackError(OpenSubdiv::Far::ErrorType err, const char *message) {
    printf("OpenSubdiv Error: %d\n", err);
    printf("    %s\n", message);
}

//------------------------------------------------------------------------------
static void
callbackModel(int m) {
    if (m < 0)
        m = 0;

    if (m >= (int)g_defaultShapes.size())
        m = (int)g_defaultShapes.size() - 1;

    g_currentShape = m;
    createOsdMesh(g_defaultShapes[g_currentShape], g_level);
}

//------------------------------------------------------------------------------
static void
callbackEndCap(int endCap) {
    g_endCap = endCap;
    createOsdMesh(g_defaultShapes[g_currentShape], g_level);
}

//------------------------------------------------------------------------------
static void
callbackKernel(int k) {

    g_kernel = k;

#ifdef OPENSUBDIV_HAS_OPENCL
    if (g_kernel == kCL && (!g_clDeviceContext.IsInitialized())) {
        if (g_clDeviceContext.Initialize() == false) {
            printf("Error in initializing OpenCL\n");
            exit(1);
        }
    }
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    if (g_kernel == kCUDA && (!g_cudaDeviceContext.IsInitialized())) {
        if (g_cudaDeviceContext.Initialize() == false) {
            printf("Error in initializing Cuda\n");
            exit(1);
        }
    }
#endif

    createOsdMesh(g_defaultShapes[g_currentShape], g_level);

}

//------------------------------------------------------------------------------
static void
callbackLevel(int l) {
    g_level = l;
    createOsdMesh(g_defaultShapes[g_currentShape], g_level);
}

//------------------------------------------------------------------------------
static void
callbackDisplayVaryingColors(int mode) {
    g_drawMode = mode;
}

//------------------------------------------------------------------------------
static void
callbackCheckBox(bool checked, int button) {
    switch (button) {
    case kHUD_CB_DISPLAY_CONTROL_MESH_EDGES:
        g_controlMeshDisplay.SetEdgesDisplay(checked);
        break;
    case kHUD_CB_DISPLAY_CONTROL_MESH_VERTS:
        g_controlMeshDisplay.SetVerticesDisplay(checked);
        break;
    case kHUD_CB_ANIMATE_VERTICES:
        g_moveScale = checked;
        break;
    case kHUD_CB_ANIMATE_PARTICLES:
        g_animParticles = checked;
        break;
    case kHUD_CB_RANDOM_START:
        g_randomStart = checked;
        createOsdMesh(g_defaultShapes[g_currentShape], g_level);
        break;
    case kHUD_CB_FREEZE:
        g_freeze = checked;
        break;
    case kHUD_CB_ADAPTIVE:
        g_adaptive = checked;
        createOsdMesh(g_defaultShapes[g_currentShape], g_level);
        break;
    case kHUD_CB_SMOOTH_CORNER_PATCH:
        g_smoothCornerPatch = checked;
        createOsdMesh(g_defaultShapes[g_currentShape], g_level);
        return;
    case kHUD_CB_SINGLE_CREASE_PATCH:
        g_singleCreasePatch = checked;
        createOsdMesh(g_defaultShapes[g_currentShape], g_level);
        return;
    case kHUD_CB_INF_SHARP_PATCH:
        g_infSharpPatch = checked;
        createOsdMesh(g_defaultShapes[g_currentShape], g_level);
        return;
    }
}

//------------------------------------------------------------------------------
static void
initHUD() {
    int windowWidth = g_width, windowHeight = g_height,
        frameBufferWidth = g_width, frameBufferHeight = g_height;

    // window size might not match framebuffer size on a high DPI display
    glfwGetWindowSize(g_window, &windowWidth, &windowHeight);
    glfwGetFramebufferSize(g_window, &frameBufferWidth, &frameBufferHeight);

    g_hud.Init(windowWidth, windowHeight, frameBufferWidth, frameBufferHeight);

    g_hud.AddCheckBox("Control edges (H)",
                      g_controlMeshDisplay.GetEdgesDisplay(),
                      10, 10, callbackCheckBox,
                      kHUD_CB_DISPLAY_CONTROL_MESH_EDGES, 'h');
    g_hud.AddCheckBox("Control vertices (J)",
                      g_controlMeshDisplay.GetVerticesDisplay(),
                      10, 30, callbackCheckBox,
                      kHUD_CB_DISPLAY_CONTROL_MESH_VERTS, 'j');
    g_hud.AddCheckBox("Animate vertices (M)", g_moveScale != 0,
                      10, 50, callbackCheckBox, kHUD_CB_ANIMATE_VERTICES, 'm');
    g_hud.AddCheckBox("Animate particles (P)", g_animParticles != 0,
                      10, 70, callbackCheckBox, kHUD_CB_ANIMATE_PARTICLES, 'p');
    g_hud.AddCheckBox("Freeze (spc)", g_freeze != 0,
                      10, 90, callbackCheckBox, kHUD_CB_FREEZE, ' ');

    g_hud.AddCheckBox("Random Start", g_randomStart,
                      10, 110, callbackCheckBox, kHUD_CB_RANDOM_START);

    int compute_pulldown = g_hud.AddPullDown("Compute (K)", 475, 10, 300,
                                             callbackKernel, 'k');
    g_hud.AddPullDownButton(compute_pulldown, "CPU", kCPU);
#ifdef OPENSUBDIV_HAS_OPENMP
    g_hud.AddPullDownButton(compute_pulldown, "OPENMP", kOPENMP);
#endif
#ifdef OPENSUBDIV_HAS_TBB
    g_hud.AddPullDownButton(compute_pulldown, "TBB", kTBB);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    g_hud.AddPullDownButton(compute_pulldown, "CUDA", kCUDA);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    if (CLDeviceContext::HAS_CL_VERSION_1_1()) {
        g_hud.AddPullDownButton(compute_pulldown, "OpenCL", kCL);
    }
#endif
#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    g_hud.AddPullDownButton(compute_pulldown, "GL XFB", kGLXFB);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    if (GLUtils::GL_ARBComputeShaderOrGL_VERSION_4_3()) {
        g_hud.AddPullDownButton(compute_pulldown, "GL Compute", kGLCompute);
    }
#endif

    int shading_pulldown = g_hud.AddPullDown("Shading (W)", 250, 10, 250, callbackDisplayVaryingColors, 'w');
    g_hud.AddPullDownButton(shading_pulldown, "(u,v)", kUV, g_drawMode==kUV);
    g_hud.AddPullDownButton(shading_pulldown, "Varying", kVARYING, g_drawMode==kVARYING);
    g_hud.AddPullDownButton(shading_pulldown, "Normal", kNORMAL, g_drawMode==kNORMAL);
    g_hud.AddPullDownButton(shading_pulldown, "Shade", kSHADE, g_drawMode==kSHADE);
    g_hud.AddPullDownButton(shading_pulldown, "FaceVarying", kFACEVARYING, g_drawMode==kFACEVARYING);
    g_hud.AddPullDownButton(shading_pulldown, "Mean Curvature", kMEAN_CURVATURE, g_drawMode==kMEAN_CURVATURE);

    g_hud.AddCheckBox("Adaptive (`)", g_adaptive != 0, 10, 150, callbackCheckBox, kHUD_CB_ADAPTIVE, '`');

    g_hud.AddCheckBox("Smooth Corner Patch (O)", g_smoothCornerPatch!=0,
                      10, 170, callbackCheckBox, kHUD_CB_SMOOTH_CORNER_PATCH, 'o');
//  g_hud.AddCheckBox("Single Crease Patch (S)", g_singleCreasePatch!=0,
//                    10, 190, callbackCheckBox, kHUD_CB_SINGLE_CREASE_PATCH, 's');
    g_hud.AddCheckBox("Inf Sharp Patch (I)", g_infSharpPatch!=0,
                      10, 190, callbackCheckBox, kHUD_CB_INF_SHARP_PATCH, 'i');

    int endcap_pulldown = g_hud.AddPullDown("End cap (E)", 10, 230, 200,
                                            callbackEndCap, 'e');
    g_hud.AddPullDownButton(endcap_pulldown, "Linear", kEndCapBilinearBasis,
                            g_endCap == kEndCapBilinearBasis);
    g_hud.AddPullDownButton(endcap_pulldown, "Regular", kEndCapBSplineBasis,
                            g_endCap == kEndCapBSplineBasis);
    g_hud.AddPullDownButton(endcap_pulldown, "Gregory", kEndCapGregoryBasis,
                            g_endCap == kEndCapGregoryBasis);

    for (int i = 1; i < 11; ++i) {
        char level[16];
        sprintf(level, "Lv. %d", i);
        g_hud.AddRadioButton(3, level, i==g_level, 10, 270+i*20, callbackLevel, i, '0'+(i%10));
    }

    int pulldown_handle = g_hud.AddPullDown("Shape (N)", -300, 10, 300, callbackModel, 'n');
    for (int i = 0; i < (int)g_defaultShapes.size(); ++i) {
        g_hud.AddPullDownButton(pulldown_handle, g_defaultShapes[i].name.c_str(),i);
    }

    g_hud.Rebuild(windowWidth, windowHeight, frameBufferWidth, frameBufferHeight);
}

//------------------------------------------------------------------------------
static void
initGL() {

    glClearColor(0.1f, 0.1f, 0.1f, 0.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);
    glGenVertexArrays(1, &g_samplesVAO);
}

//------------------------------------------------------------------------------
static void
uninitGL() {
    glDeleteVertexArrays(1, &g_samplesVAO);
}

//------------------------------------------------------------------------------
static void
callbackErrorGLFW(int error, const char* description) {
    fprintf(stderr, "GLFW Error (%d) : %s\n", error, description);
}

//------------------------------------------------------------------------------
int main(int argc, char **argv) {

    ArgOptions args;

    args.Parse(argc, argv);
    args.PrintUnrecognizedArgsWarnings();

    g_yup = args.GetYUp();
    g_adaptive = args.GetAdaptive();
    g_level = args.GetLevel();
    g_repeatCount = args.GetRepeatCount();

    ViewerArgsUtils::PopulateShapes(args, &g_defaultShapes);

    initShapes();

    Far::SetErrorCallback(callbackError);

    glfwSetErrorCallback(callbackErrorGLFW);
    if (! glfwInit()) {
        printf("Failed to initialize GLFW\n");
        return 1;
    }

    static const char windowTitle[] = "OpenSubdiv glEvalLimit " OPENSUBDIV_VERSION_STRING;

    GLUtils::SetMinimumGLVersion();

    if (args.GetFullScreen()) {

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
             args.GetFullScreen() && g_primary ? g_primary : NULL, NULL))) {
        std::cerr << "Failed to create OpenGL context.\n";
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(g_window);

    GLUtils::InitializeGL();
    GLUtils::PrintGLVersion();

    // accommodate high DPI displays (e.g. mac retina displays)
    glfwGetFramebufferSize(g_window, &g_width, &g_height);
    glfwSetFramebufferSizeCallback(g_window, reshape);

    glfwSetKeyCallback(g_window, keyboard);
    glfwSetCursorPosCallback(g_window, motion);
    glfwSetMouseButtonCallback(g_window, mouse);
    glfwSetWindowCloseCallback(g_window, windowClose);
    initGL();
    linkDefaultProgram();

    glfwSwapInterval(0);

    initHUD();
    callbackModel(g_currentShape);

    g_fpsTimer.Start();

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
