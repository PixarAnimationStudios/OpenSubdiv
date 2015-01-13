//
//   Copyright 2014 Pixar
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

#if defined(__APPLE__)
    #if defined(OSD_USES_GLEW)
        #include <GL/glew.h>
    #else
        #include <OpenGL/gl3.h>
    #endif
    #define GLFW_INCLUDE_GL3
    #define GLFW_NO_GLU
#else
    #include <stdlib.h>
    #include <GL/glew.h>
    #if defined(WIN32)
        #include <GL/wglew.h>
    #endif
#endif

#include <GLFW/glfw3.h>
GLFWwindow* g_window=0;
GLFWmonitor* g_primary=0;

#include <osd/vertex.h>
#include <osd/glDrawContext.h>
#include <osd/glDrawRegistry.h>
#include <osd/glMesh.h>
#include <far/error.h>

#include <osd/cpuGLVertexBuffer.h>
#include <osd/cpuComputeContext.h>
#include <osd/cpuComputeController.h>

#ifdef OPENSUBDIV_HAS_OPENMP
    #include <osd/ompComputeController.h>
#endif

#ifdef OPENSUBDIV_HAS_TBB
    #include <osd/tbbComputeController.h>
#endif

#ifdef OPENSUBDIV_HAS_OPENCL
    #include <osd/clGLVertexBuffer.h>
    #include <osd/clComputeContext.h>
    #include <osd/clComputeController.h>

    #include "../common/clInit.h"

    cl_context g_clContext;
    cl_command_queue g_clQueue;
#endif

#ifdef OPENSUBDIV_HAS_CUDA
    #include <osd/cudaGLVertexBuffer.h>
    #include <osd/cudaComputeContext.h>
    #include <osd/cudaComputeController.h>

    #include <cuda_runtime_api.h>
    #include <cuda_gl_interop.h>

    #include "../common/cudaInit.h"
    bool g_cudaInitialized = false;
#endif

#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    #include <osd/glslTransformFeedbackComputeContext.h>
    #include <osd/glslTransformFeedbackComputeController.h>
    #include <osd/glVertexBuffer.h>
#endif

#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    #include <osd/glslComputeContext.h>
    #include <osd/glslComputeController.h>
    #include <osd/glVertexBuffer.h>
#endif


#include <common/vtr_utils.h>
#include <shapes/catmark_cube.h>
#include <shapes/catmark_bishop.h>
#include <shapes/catmark_pawn.h>
#include <shapes/catmark_rook.h>

#include "../common/stopwatch.h"
#include "../common/simple_math.h"
#include "../common/gl_hud.h"

static const char *shaderSource =
#include "shader.gen.h"
;

#include <cfloat>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace OpenSubdiv;

// ---------------------------------------------------------------------------

class InstancesBase {
public:
    virtual ~InstancesBase() {}

    virtual void UpdateVertexBuffer(int instance, std::vector<float> const &src) = 0;
    virtual void UpdateVaryingBuffer(int instance, std::vector<float> const &src) = 0;

    virtual GLuint BindVertexBuffer() = 0;
    virtual GLuint BindVaryingBuffer() = 0;

    Osd::VertexBufferDescriptor const &GetVertexDesc() const {
        return _vertexDesc;
    }
    Osd::VertexBufferDescriptor const &GetVaryingDesc() const {
        return _varyingDesc;
    }

protected:
    InstancesBase(Osd::VertexBufferDescriptor const &vertexDesc,
                  Osd::VertexBufferDescriptor const &varyingDesc,
                  int numVertices) :
        _vertexDesc(vertexDesc),
        _varyingDesc(varyingDesc),
        _numVertices(numVertices) {
    }

    int getNumVertices() const { return _numVertices; }

private:
    Osd::VertexBufferDescriptor _vertexDesc;
    Osd::VertexBufferDescriptor _varyingDesc;
    int _numVertices;                // # of vertices of single instance
};

template <class VERTEX_BUFFER>
class Instances : public InstancesBase {
public:
    Instances(int numInstances,
              Osd::VertexBufferDescriptor const &vertexDesc,
              Osd::VertexBufferDescriptor const &varyingDesc,
              bool interleaved,
              int numVertices) :
        InstancesBase(vertexDesc, varyingDesc, numVertices),
        _vertexBuffer(NULL), _varyingBuffer(NULL), _interleaved(interleaved) {

        if (interleaved) {
            assert(vertexDesc.stride == varyingDesc.stride);

            _vertexBuffer = createVertexBuffer(
                vertexDesc.stride, numInstances * numVertices);
        } else {
            if (vertexDesc.stride > 0) {
                _vertexBuffer = createVertexBuffer(
                    vertexDesc.stride, numInstances * numVertices);
            }
            if (varyingDesc.stride > 0) {
                _varyingBuffer = createVertexBuffer(
                    varyingDesc.stride, numInstances * numVertices);
            }
        }
    }

    virtual ~Instances() {
        delete _vertexBuffer;
        delete _varyingBuffer;
    }

    virtual void UpdateVertexBuffer(int instance, std::vector<float> const &src) {
        updateVertexBuffer(_vertexBuffer, &src[0], instance * getNumVertices(),
                           (int)src.size()/_vertexBuffer->GetNumElements());
    }
    virtual void UpdateVaryingBuffer(int instance, std::vector<float> const &src) {
        updateVertexBuffer(_varyingBuffer, &src[0], instance * getNumVertices(),
                           (int)src.size()/_varyingBuffer->GetNumElements());
    }

    virtual GLuint BindVertexBuffer() {
        return _vertexBuffer->BindVBO();
    }

    virtual GLuint BindVaryingBuffer() {
        return _varyingBuffer->BindVBO();
    }

    VERTEX_BUFFER *createVertexBuffer(int numElements, int numVertices) {
        return VERTEX_BUFFER::Create(numElements, numVertices);
    }
    void updateVertexBuffer(VERTEX_BUFFER *vertexBuffer, const float *src, int startVertex,
                            int numVertices) {
        vertexBuffer->UpdateData(src, startVertex, numVertices);
    }

    VERTEX_BUFFER *GetVertexBuffer() const { return _vertexBuffer; }
    VERTEX_BUFFER *GetVaryingBuffer() const { return _interleaved ? _vertexBuffer :_varyingBuffer; }

private:
    VERTEX_BUFFER *_vertexBuffer;
    VERTEX_BUFFER *_varyingBuffer;
    bool _interleaved;
};

// ---------------------------------------------------------------------------

class TopologyBase {
public:
    virtual ~TopologyBase() {
        delete _drawContext;
    }

    virtual void Refine(InstancesBase *instance, int numInstances) = 0;

    virtual InstancesBase *CreateInstances(
        int numInstances,
        Osd::VertexBufferDescriptor const &vertexDesc,
        Osd::VertexBufferDescriptor const &varyingDesc,
        bool interleaved) = 0;

    virtual void UpdateVertexTexture(InstancesBase *instances) = 0;

    virtual void Synchronize() = 0;

    Osd::GLDrawContext *GetDrawContext() const {
        return _drawContext;
    }

    void SetRestPosition(std::vector<float> const &restPosition) {
        _restPosition = restPosition;
    }

    std::vector<float> const &GetRestPosition() const {
        return _restPosition;
    }

    int GetNumVertices() const {
        return _numVertices;
    }

protected:

    TopologyBase(Far::PatchTables const * patchTables) {
        _drawContext = Osd::GLDrawContext::Create(patchTables, 7);
    }

    void updateVertexBufferStride(int stride) {
        // modifying patchArrays in drawcontext.
        Osd::DrawContext::PatchArrayVector &patchArrays =
            _drawContext->GetPatchArrays();
        for (int i = 0; i < (int)patchArrays.size(); ++i) {
            Osd::DrawContext::PatchDescriptor desc = patchArrays[i].GetDescriptor();
            desc.SetNumElements(stride);
            patchArrays[i].SetDescriptor(desc);
        }
    }

    int _numVertices;

private:
    Osd::GLDrawContext *_drawContext;
    std::vector<float> _restPosition;
};

template <class COMPUTE_CONTROLLER, class VERTEX_BUFFER>
class Topology : public TopologyBase {

public:

    typedef typename COMPUTE_CONTROLLER::ComputeContext ComputeContext;

    Topology(Far::PatchTables const * patchTables,
        Far::StencilTables const * vertexStencils,
            Far::StencilTables const * varyingStencils)
                : TopologyBase(patchTables) {

        _computeContext = ComputeContext::Create(vertexStencils, varyingStencils);

        _kernelBatches.push_back(Far::StencilTablesFactory::Create(*vertexStencils));

        _numVertices = vertexStencils->GetNumStencils() +
            vertexStencils->GetNumControlVertices();
    }

    ~Topology() {
        delete _computeContext;
    }

    void Refine(InstancesBase *instance, int numInstances) {

        Osd::VertexBufferDescriptor const &globalVertexDesc =
            instance->GetVertexDesc();
        Osd::VertexBufferDescriptor const &globalVaryingDesc =
            instance->GetVaryingDesc();

        Instances<VERTEX_BUFFER> *typedInstance =
            static_cast<Instances<VERTEX_BUFFER> *>(instance);

        for (int i = 0; i < numInstances; ++i) {

            Osd::VertexBufferDescriptor vertexDesc(
                globalVertexDesc.offset + _numVertices*globalVertexDesc.stride*i,
                globalVertexDesc.length,
                globalVertexDesc.stride);

            Osd::VertexBufferDescriptor varyingDesc(
                globalVaryingDesc.offset + _numVertices*globalVaryingDesc.stride*i,
                globalVaryingDesc.length,
                globalVaryingDesc.stride);

            _computeController.Compute(_computeContext,
                                      _kernelBatches,
                                      typedInstance->GetVertexBuffer(),
                                      typedInstance->GetVaryingBuffer(),
                                      &vertexDesc,
                                      &varyingDesc);
        }
    }

    virtual InstancesBase *CreateInstances(
        int numInstances,
        Osd::VertexBufferDescriptor const &vertexDesc,
        Osd::VertexBufferDescriptor const &varyingDesc,
        bool interleaved) {

        return new Instances<VERTEX_BUFFER>(numInstances,
                                            vertexDesc,
                                            varyingDesc,
                                            interleaved,
                                            _numVertices);
    }

    virtual void Synchronize() {
        _computeController.Synchronize();
    }

    virtual void UpdateVertexTexture(InstancesBase *instances) {
        Instances<VERTEX_BUFFER> *typedInstance =
            static_cast<Instances<VERTEX_BUFFER> *>(instances);
        GetDrawContext()->UpdateVertexTexture(typedInstance->GetVertexBuffer());

        updateVertexBufferStride(typedInstance->GetVertexBuffer()->GetNumElements());
    }

private:
    COMPUTE_CONTROLLER _computeController;
    ComputeContext *_computeContext;
    Far::KernelBatchVector _kernelBatches;
};

// ---------------------------------------------------------------------------

// CL specializations
#ifdef OPENSUBDIV_HAS_OPENCL

template<> Osd::CLGLVertexBuffer *
Instances<Osd::CLGLVertexBuffer>::createVertexBuffer(
    int numElements, int numVertices) {
    return Osd::CLGLVertexBuffer::Create(
        numElements, numVertices, g_clContext);
}

template<> void
Instances<Osd::CLGLVertexBuffer>::updateVertexBuffer(
    Osd::CLGLVertexBuffer *vertexBuffer,
    const float *src, int startVertex, int numVertices) {
    vertexBuffer->UpdateData(src, startVertex, numVertices, g_clQueue);
}

template<>
Topology<Osd::CLComputeController, Osd::CLGLVertexBuffer>::
Topology(Far::PatchTables const * patchTables,
    Far::StencilTables const * vertexStencils, Far::StencilTables const * varyingStencils) :
        TopologyBase(patchTables), _computeController(g_clContext, g_clQueue) {

    _computeContext = ComputeContext::Create(g_clContext, vertexStencils, varyingStencils);

    _kernelBatches.push_back(Far::StencilTablesFactory::Create(*vertexStencils));

    _numVertices = vertexStencils->GetNumStencils() +
        vertexStencils->GetNumControlVertices();
}
#endif

// ---------------------------------------------------------------------------


TopologyBase *g_topology = NULL;
InstancesBase *g_instances = NULL;

enum KernelType { kCPU = 0,
                  kOPENMP = 1,
                  kTBB = 2,
                  kCUDA = 3,
                  kCL = 4,
                  kGLSL = 5,
                  kGLSLCompute = 6 };

enum DisplayStyle { kWire = 0,
                    kShaded,
                    kWireShaded,
                    kVarying,
                    kVaryingInterleaved };

enum HudCheckBox { kHUD_CB_FREEZE };

// GUI variables
int   g_displayStyle = kShaded,
      g_adaptive = 0,
      g_mbutton[3] = {0, 0, 0},
      g_freeze = 0,
      g_running = 1;

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

// performance
float g_cpuTime = 0;
float g_gpuTime = 0;
Stopwatch g_fpsTimer;

int g_level = 2;
int g_tessLevel = 1;
int g_tessLevelMin = 1;
int g_numInstances = 25;
int g_frame = 0;
int g_kernel = kCPU;

GLuint g_transformUB = 0,
       g_transformBinding = 0,
       g_tessellationUB = 0,
       g_tessellationBinding = 0,
       g_lightingUB = 0,
       g_lightingBinding = 0;

struct Transform {
    float ModelViewMatrix[16];
    float ProjectionMatrix[16];
    float ModelViewProjectionMatrix[16];
} g_transformData;

GLuint g_queries[2] = {0, 0};
GLuint g_vao = 0;

static void
checkGLErrors(std::string const & where = "") {
    GLuint err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::cerr << "GL error: "
                  << (where.empty() ? "" : where + " ")
                  << err << "\n";
    }
}

//------------------------------------------------------------------------------
struct SimpleShape {
    std::string  name;
    Scheme       scheme;
    std::string  data;

    SimpleShape() { }
    SimpleShape( std::string const & idata, char const * iname, Scheme ischeme )
        : name(iname), scheme(ischeme), data(idata) { }
};

//------------------------------------------------------------------------------
static void
updateGeom() {

    std::vector<float> const &restPosition = g_topology->GetRestPosition();

    int nverts = (int)restPosition.size()/3;
    int numVertexElements = (g_displayStyle == kVaryingInterleaved ? 7 : 3);
    int numVaryingElements = (g_displayStyle == kVarying ? 4 : 0);

    std::vector<float> vertex(numVertexElements * nverts);
    std::vector<float> varying(numVaryingElements * nverts);

    int column = (int)ceil(sqrt((float)g_numInstances));
    for (int i = 0; i < g_numInstances; ++i) {
        float *d = &vertex[0];
        const float *p = &restPosition[0];

        for (int j = 0; j < nverts; ++j) {
            *d++ = p[0] + i%column - 0.5f*(column-1);
            *d++ = p[1] + i/column - 0.5f*(column-1);
            *d++ = p[2] * (float)(1+sin(0.1f*g_frame + i));
            p += 3;

            if (g_displayStyle == kVaryingInterleaved) {
                *d++ = (1+(float)sin(0.1f*g_frame + i)) * 0.5f;
                *d++ = 1;
                *d++ = 1;
                *d++ = 1.0;
            }
        }
        g_instances->UpdateVertexBuffer(i, vertex);

        if (g_displayStyle == kVarying) {
            float *d = &varying[0];
            for (int j = 0; j < nverts; ++j) {
                *d++ = 1;
                *d++ = (1+(float)sin(0.1f*g_frame + i)) * 0.5f;
                *d++ = 1;
                *d++ = 1.0;
            }
            g_instances->UpdateVaryingBuffer(i, varying);
        }
    }
}

static void
refine() {

    Stopwatch s;
    s.Start();

    g_topology->Refine(g_instances, g_numInstances);

    s.Stop();
    g_cpuTime = float(s.GetElapsed() * 1000.0f);
    s.Start();

    g_topology->Synchronize();

    s.Stop();
    g_gpuTime = float(s.GetElapsed() * 1000.0f);


    s.Stop();
}

//------------------------------------------------------------------------------
static TopologyBase *
createOsdMesh( const std::string &shapeStr, int level, Scheme scheme=kCatmark ) {

    checkGLErrors("create osd enter");

    Shape * shape = Shape::parseObj(shapeStr.c_str(), scheme);

    std::vector<float> restPosition(shape->verts);

    Far::TopologyRefiner * refiner = 0;
    {
        Sdc::SchemeType type = GetSdcType(*shape);
        Sdc::Options options = GetSdcOptions(*shape);

        refiner = Far::TopologyRefinerFactory<Shape>::Create(*shape,
                    Far::TopologyRefinerFactory<Shape>::Options(type, options));

        assert(refiner);
    }

    // material assignment
    std::vector<int> idsOnPtexFaces;
    {
        int numFaces = refiner->GetNumFaces(0);

        // first, assign material ID to each coarse face
        std::vector<int> idsOnCoarseFaces;
        for (int i = 0; i < numFaces; ++i) {
            int materialID = i%6;
            idsOnCoarseFaces.push_back(materialID);
        }

        // create ptex index to coarse face index mapping
        int numPtexFaces = refiner->GetNumPtexFaces();

        // XXX: duped logic to simpleHbr
        std::vector<int> ptexIndexToFaceMapping(numPtexFaces);
        int ptexIndex = 0;
        for (int face=0; face < numFaces; ++face) {

            ptexIndexToFaceMapping[ptexIndex++] = face;
            Far::ConstIndexArray fverts = refiner->GetFaceVertices(0, face);
            if ( (scheme==kCatmark or scheme==kBilinear) and fverts.size() != 4 ) {
                for (int j = 0; j < (fverts.size()-1); ++j) {
                    ptexIndexToFaceMapping[ptexIndex++] = face;
                }
            }
        }

        // convert ID array from coarse face index space to ptex index space
        for (int i = 0; i < numPtexFaces; ++i) {
            idsOnPtexFaces.push_back(idsOnCoarseFaces[ptexIndexToFaceMapping[i]]);
        }
    }

    // Adaptive refinement currently supported only for catmull-clark scheme
    bool doAdaptive = (g_adaptive!=0 and scheme==kCatmark);

    if (doAdaptive) {
        refiner->RefineAdaptive(Far::TopologyRefiner::AdaptiveOptions(level));
    } else {
        Far::TopologyRefiner::UniformOptions options(level);
        options.fullTopologyInLastLevel = true;
        refiner->RefineUniform(options);
    }

    Far::StencilTables const * vertexStencils=0, * varyingStencils=0;
    {
        Far::StencilTablesFactory::Options options;
        options.generateOffsets = true;
        options.generateIntermediateLevels = doAdaptive ? true : false;

        vertexStencils = Far::StencilTablesFactory::Create(*refiner, options);

        if (g_displayStyle==kVarying or g_displayStyle==kVaryingInterleaved) {
            varyingStencils = Far::StencilTablesFactory::Create(*refiner, options);
        }

        assert(vertexStencils);
    }

    Far::PatchTables const * patchTables =
        Far::PatchTablesFactory::Create(*refiner);


    // create partitioned patcharray
    TopologyBase *topology = NULL;

    if (g_kernel == kCPU) {
        topology = new Topology<Osd::CpuComputeController,
            Osd::CpuGLVertexBuffer>(patchTables, vertexStencils, varyingStencils);
#ifdef OPENSUBDIV_HAS_OPENMP
    } else if (g_kernel == kOPENMP) {
        topology = new Topology<Osd::OmpComputeController,
            Osd::CpuGLVertexBuffer>(patchTables, vertexStencils, varyingStencils);
#endif
#ifdef OPENSUBDIV_HAS_TBB
    } else if (g_kernel == kTBB) {
        topology = new Topology<Osd::TbbComputeController,
            Osd::CpuGLVertexBuffer>(patchTables, vertexStencils, varyingStencils);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    } else if (g_kernel == kCUDA) {
        topology = new Topology<Osd::CudaComputeController,
            Osd::CudaGLVertexBuffer>(patchTables, vertexStencils, varyingStencils);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    } else if (g_kernel == kCL) {
        topology = new Topology<Osd::CLComputeController,
            Osd::CLGLVertexBuffer>(patchTables, vertexStencils, varyingStencils);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    } else if (g_kernel == kGLSL) {
        topology = new Topology<Osd::GLSLTransformFeedbackComputeController,
            Osd::GLVertexBuffer>(patchTables, vertexStencils, varyingStencils);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    } else if (g_kernel == kGLSLCompute) {
        topology = new Topology<Osd::GLSLComputeController,
            Osd::GLVertexBuffer>(patchTables, vertexStencils, varyingStencils);
#endif
    } else {
    }

    delete refiner;
    delete vertexStencils;
    delete varyingStencils;
    delete patchTables;

    // centering rest position
    float min[3] = { FLT_MAX,  FLT_MAX,  FLT_MAX};
    float max[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    float center[3];
    for (size_t i=0; i < restPosition.size()/3; ++i) {
        for (int j=0; j<3; ++j) {
            float v = restPosition[i*3+j];
            min[j] = std::min(min[j], v);
            max[j] = std::max(max[j], v);
        }
    }
    for (int j=0; j<3; ++j) center[j] = (min[j] + max[j]) * 0.5f;
    for (size_t i=0; i < restPosition.size()/3; ++i) {
        restPosition[i*3+0] -= center[0];
        restPosition[i*3+1] -= center[1];
        restPosition[i*3+2] -= min[2];
    }

    // save rest position
    topology->SetRestPosition(restPosition);

    return topology;
}

//------------------------------------------------------------------------------
static void
fitFrame() {

    g_pan[0] = g_pan[1] = 0;
    g_dolly = g_size;
}

//------------------------------------------------------------------------------

union Effect {
    Effect(int displayStyle_) : value(0) {
        displayStyle = displayStyle_;
    }

    struct {
        unsigned int displayStyle:3;
    };
    int value;

    bool operator < (const Effect &e) const {
        return value < e.value;
    }
};

typedef std::pair<Osd::DrawContext::PatchDescriptor, Effect> EffectDesc;

class EffectDrawRegistry : public Osd::GLDrawRegistry<EffectDesc> {

protected:
    virtual ConfigType *
    _CreateDrawConfig(DescType const & desc, SourceConfigType const * sconfig);

    virtual SourceConfigType *
    _CreateDrawSourceConfig(DescType const & desc);
};

EffectDrawRegistry::SourceConfigType *
EffectDrawRegistry::_CreateDrawSourceConfig(DescType const & desc) {

    Effect effect = desc.second;

    SourceConfigType * sconfig =
        BaseRegistry::_CreateDrawSourceConfig(desc.first);

    assert(sconfig);

#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
    const char *glslVersion = "#version 400\n";
#else
    const char *glslVersion = "#version 330\n";
#endif

    if (desc.first.GetType() == Far::PatchDescriptor::QUADS or
        desc.first.GetType() == Far::PatchDescriptor::TRIANGLES) {
        sconfig->vertexShader.source = shaderSource;
        sconfig->vertexShader.version = glslVersion;
        sconfig->vertexShader.AddDefine("VERTEX_SHADER");
    } else {
        sconfig->geometryShader.AddDefine("SMOOTH_NORMALS");
    }

    sconfig->geometryShader.source = shaderSource;
    sconfig->geometryShader.version = glslVersion;
    sconfig->geometryShader.AddDefine("GEOMETRY_SHADER");

    sconfig->fragmentShader.source = shaderSource;
    sconfig->fragmentShader.version = glslVersion;
    sconfig->fragmentShader.AddDefine("FRAGMENT_SHADER");

    if (desc.first.GetType() == Far::PatchDescriptor::QUADS) {
        // uniform catmark, bilinear
        sconfig->geometryShader.AddDefine("PRIM_QUAD");
        sconfig->fragmentShader.AddDefine("PRIM_QUAD");
        sconfig->commonShader.AddDefine("UNIFORM_SUBDIVISION");
    } else if (desc.first.GetType() == Far::PatchDescriptor::TRIANGLES) {
        // uniform loop
        sconfig->geometryShader.AddDefine("PRIM_TRI");
        sconfig->fragmentShader.AddDefine("PRIM_TRI");
        sconfig->commonShader.AddDefine("LOOP");
        sconfig->commonShader.AddDefine("UNIFORM_SUBDIVISION");
    } else {
        // adaptive
        sconfig->vertexShader.source = shaderSource + sconfig->vertexShader.source;
        sconfig->tessControlShader.source = shaderSource + sconfig->tessControlShader.source;
        sconfig->tessEvalShader.source = shaderSource + sconfig->tessEvalShader.source;

        sconfig->geometryShader.AddDefine("PRIM_TRI");
        sconfig->fragmentShader.AddDefine("PRIM_TRI");
    }

    switch (effect.displayStyle) {
        case kWire:
            sconfig->commonShader.AddDefine("GEOMETRY_OUT_WIRE");
            break;
        case kWireShaded:
            sconfig->commonShader.AddDefine("GEOMETRY_OUT_LINE");
            break;
        case kShaded:
            sconfig->commonShader.AddDefine("GEOMETRY_OUT_FILL");
            break;
        case kVarying:
            sconfig->commonShader.AddDefine("VARYING_COLOR");
            sconfig->commonShader.AddDefine("GEOMETRY_OUT_FILL");
            break;
        case kVaryingInterleaved:
            sconfig->commonShader.AddDefine("VARYING_COLOR");
            sconfig->commonShader.AddDefine("GEOMETRY_OUT_FILL");
            break;
    }

    return sconfig;
}

EffectDrawRegistry::ConfigType *
EffectDrawRegistry::_CreateDrawConfig(
        DescType const & desc,
        SourceConfigType const * sconfig) {

    ConfigType * config = BaseRegistry::_CreateDrawConfig(desc.first, sconfig);
    assert(config);

    GLuint uboIndex;

    // XXXdyu can use layout(binding=) with GLSL 4.20 and beyond
    g_transformBinding = 0;
    uboIndex = glGetUniformBlockIndex(config->program, "Transform");
    if (uboIndex != GL_INVALID_INDEX)
        glUniformBlockBinding(config->program, uboIndex, g_transformBinding);

    g_tessellationBinding = 1;
    uboIndex = glGetUniformBlockIndex(config->program, "Tessellation");
    if (uboIndex != GL_INVALID_INDEX)
        glUniformBlockBinding(config->program, uboIndex, g_tessellationBinding);

    g_lightingBinding = 2;
    uboIndex = glGetUniformBlockIndex(config->program, "Lighting");
    if (uboIndex != GL_INVALID_INDEX)
        glUniformBlockBinding(config->program, uboIndex, g_lightingBinding);

    GLint loc;
#if not defined(GL_ARB_separate_shader_objects) || defined(GL_VERSION_4_1)
    glUseProgram(config->program);
    if ((loc = glGetUniformLocation(config->program, "OsdVertexBuffer")) != -1) {
        glUniform1i(loc, 0); // GL_TEXTURE0
    }
    if ((loc = glGetUniformLocation(config->program, "OsdValenceBuffer")) != -1) {
        glUniform1i(loc, 1); // GL_TEXTURE1
    }
    if ((loc = glGetUniformLocation(config->program, "OsdQuadOffsetBuffer")) != -1) {
        glUniform1i(loc, 2); // GL_TEXTURE2
    }
    if ((loc = glGetUniformLocation(config->program, "OsdPatchParamBuffer")) != -1) {
        glUniform1i(loc, 3); // GL_TEXTURE3
    }
    if ((loc = glGetUniformLocation(config->program, "OsdFVarDataBuffer")) != -1) {
        glUniform1i(loc, 4); // GL_TEXTURE4
    }
#else
    if ((loc = glGetUniformLocation(config->program, "OsdVertexBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 0); // GL_TEXTURE0
    }
    if ((loc = glGetUniformLocation(config->program, "OsdValenceBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 1); // GL_TEXTURE1
    }
    if ((loc = glGetUniformLocation(config->program, "OsdQuadOffsetBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 2); // GL_TEXTURE2
    }
    if ((loc = glGetUniformLocation(config->program, "OsdPatchParamBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 3); // GL_TEXTURE3
    }
    if ((loc = glGetUniformLocation(config->program, "OsdFVarDataBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 4); // GL_TEXTURE4
    }
#endif

    return config;
}

EffectDrawRegistry effectRegistry;

static Effect
GetEffect() {

    return Effect(g_displayStyle);
}

//------------------------------------------------------------------------------
static GLuint
bindProgram(Effect effect, Osd::DrawContext::PatchArray const & patch) {

    EffectDesc effectDesc(patch.GetDescriptor(), effect);
    EffectDrawRegistry::ConfigType *
        config = effectRegistry.GetDrawConfig(effectDesc);

    GLuint program = config->program;

    glUseProgram(program);

    if (! g_transformUB) {
        glGenBuffers(1, &g_transformUB);
        glBindBuffer(GL_UNIFORM_BUFFER, g_transformUB);
        glBufferData(GL_UNIFORM_BUFFER,
                sizeof(g_transformData), NULL, GL_STATIC_DRAW);
    };
    glBindBuffer(GL_UNIFORM_BUFFER, g_transformUB);
    glBufferSubData(GL_UNIFORM_BUFFER,
                0, sizeof(g_transformData), &g_transformData);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glBindBufferBase(GL_UNIFORM_BUFFER, g_transformBinding, g_transformUB);

    // Update and bind tessellation state
    struct Tessellation {
        float TessLevel;
    } tessellationData;

    tessellationData.TessLevel = static_cast<float>(1 << g_tessLevel);

    if (! g_tessellationUB) {
        glGenBuffers(1, &g_tessellationUB);
        glBindBuffer(GL_UNIFORM_BUFFER, g_tessellationUB);
        glBufferData(GL_UNIFORM_BUFFER,
                sizeof(tessellationData), NULL, GL_STATIC_DRAW);
    };
    glBindBuffer(GL_UNIFORM_BUFFER, g_tessellationUB);
    glBufferSubData(GL_UNIFORM_BUFFER,
                0, sizeof(tessellationData), &tessellationData);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glBindBufferBase(GL_UNIFORM_BUFFER, g_tessellationBinding, g_tessellationUB);

    // Update and bind lighting state
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
        glBufferData(GL_UNIFORM_BUFFER,
                sizeof(lightingData), NULL, GL_STATIC_DRAW);
    };
    glBindBuffer(GL_UNIFORM_BUFFER, g_lightingUB);
    glBufferSubData(GL_UNIFORM_BUFFER,
                0, sizeof(lightingData), &lightingData);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glBindBufferBase(GL_UNIFORM_BUFFER, g_lightingBinding, g_lightingUB);

    Osd::GLDrawContext *drawContext = g_topology->GetDrawContext();

    if (drawContext->GetVertexTextureBuffer()) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_BUFFER,
            drawContext->GetVertexTextureBuffer());
    }
    if (drawContext->GetVertexValenceTextureBuffer()) {
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_BUFFER,
            drawContext->GetVertexValenceTextureBuffer());
    }
    if (drawContext->GetQuadOffsetsTextureBuffer()) {
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_BUFFER,
            drawContext->GetQuadOffsetsTextureBuffer());
    }
    if (drawContext->GetPatchParamTextureBuffer()) {
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_BUFFER,
            drawContext->GetPatchParamTextureBuffer());
    }
    if (drawContext->GetFvarDataTextureBuffer()) {
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_BUFFER,
            drawContext->GetFvarDataTextureBuffer());
    }

    glActiveTexture(GL_TEXTURE0);

    return program;
}

//------------------------------------------------------------------------------
static int
drawPatches(Osd::DrawContext::PatchArrayVector const &patches,
            int instanceIndex,
            GLfloat const *color) {

    int numDrawCalls = 0;
    for (int i=0; i<(int)patches.size(); ++i) {

        Osd::DrawContext::PatchArray const & patch = patches[i];

        Osd::DrawContext::PatchDescriptor desc = patch.GetDescriptor();
        Far::PatchDescriptor::Type patchType = desc.GetType();

        GLenum primType;

        switch(patchType) {
        case Far::PatchDescriptor::QUADS:
            primType = GL_LINES_ADJACENCY;
            break;
        case Far::PatchDescriptor::TRIANGLES:
            primType = GL_TRIANGLES;
            break;
        default:
#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
            primType = GL_PATCHES;
            glPatchParameteri(GL_PATCH_VERTICES, desc.GetNumControlVertices());
#else
            primType = GL_POINTS;
#endif
        }

#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
        GLuint program = bindProgram(GetEffect(), patch);

        GLuint uniformColor =
          glGetUniformLocation(program, "diffuseColor");

        glProgramUniform4f(program, uniformColor, color[0], color[1], color[2], 1);

        GLuint uniformGregoryQuadOffsetBase =
          glGetUniformLocation(program, "GregoryQuadOffsetBase");
        GLuint uniformPrimitiveIdBase =
          glGetUniformLocation(program, "PrimitiveIdBase");

        glProgramUniform1i(program, uniformGregoryQuadOffsetBase,
                           patch.GetQuadOffsetIndex());
        glProgramUniform1i(program, uniformPrimitiveIdBase,
                           patch.GetPatchIndex());
#else
        GLuint program = bindProgram(GetEffect(), patch);
        GLint uniformPrimitiveIdBase =
          glGetUniformLocation(program, "PrimitiveIdBase");
        if (uniformPrimitiveIdBase != -1)
            glUniform1i(uniformPrimitiveIdBase, patch.GetPatchIndex());
#endif

        GLvoid *indices = (void *)(patch.GetVertIndex() * sizeof(unsigned int));

        int baseVertex = g_topology->GetNumVertices() * instanceIndex;

        glProgramUniform1i(program, glGetUniformLocation(program, "BaseVertex"),
                           baseVertex);

        glDrawElementsBaseVertex(primType,
                                 patch.GetNumIndices(),
                                 GL_UNSIGNED_INT,
                                 indices,
                                 baseVertex);
        ++numDrawCalls;
    }
    return numDrawCalls;
}

//------------------------------------------------------------------------------
static void
display() {

    g_hud.GetFrameBuffer()->Bind();

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
                45.0f, (float)aspect, 0.01f, 500.0f);
    multMatrix(g_transformData.ModelViewProjectionMatrix,
               g_transformData.ModelViewMatrix,
               g_transformData.ProjectionMatrix);

    glEnable(GL_DEPTH_TEST);

    // make sure that the vertex buffer is interoped back as a GL resources.
    g_instances->BindVertexBuffer();

    glBindVertexArray(g_vao);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,
                 g_topology->GetDrawContext()->GetPatchIndexBuffer());

    if (g_displayStyle == kVarying) {

        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, g_instances->BindVertexBuffer());
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 3, 0);

        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, g_instances->BindVaryingBuffer());
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 4, 0);

    } else if (g_displayStyle == kVaryingInterleaved) {

        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, g_instances->BindVertexBuffer());
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 7, 0);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 7,
                              (void*)(sizeof(GLfloat)*3));

    } else {

        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, g_instances->BindVertexBuffer());
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 3, 0);
        glDisableVertexAttribArray(1);
    }


    // update vertex buffer to texture for gregory patch drawing.
    g_topology->UpdateVertexTexture(g_instances);

    Osd::DrawContext::PatchArrayVector const & patches =
        g_topology->GetDrawContext()->GetPatchArrays();
    int numDrawCalls = 0;
    // primitive counting
    glBeginQuery(GL_PRIMITIVES_GENERATED, g_queries[0]);
#if defined(GL_VERSION_3_3)
    glBeginQuery(GL_TIME_ELAPSED, g_queries[1]);
#endif

    // draw instances with same topology
    for (int i = 0; i < g_numInstances; ++i) {
        GLfloat color[3] = {i/(float)g_numInstances, 0.5, 0.5};
        numDrawCalls += drawPatches(patches, i, color);
    }

    glEndQuery(GL_PRIMITIVES_GENERATED);
#if defined(GL_VERSION_3_3)
    glEndQuery(GL_TIME_ELAPSED);
#endif

    glBindVertexArray(0);

    glUseProgram(0);

    s.Stop();
    float drawCpuTime = float(s.GetElapsed() * 1000.0f);

    GLuint numPrimsGenerated = 0;
    GLuint timeElapsed = 0;
    glGetQueryObjectuiv(g_queries[0], GL_QUERY_RESULT, &numPrimsGenerated);
#if defined(GL_VERSION_3_3)
    glGetQueryObjectuiv(g_queries[1], GL_QUERY_RESULT, &timeElapsed);
#endif
    float drawGpuTime = timeElapsed / 1000.0f / 1000.0f;

    g_hud.GetFrameBuffer()->ApplyImageShader();

    if (g_hud.IsVisible()) {
        g_fpsTimer.Stop();
        double fps = 1.0/g_fpsTimer.GetElapsed();
        g_fpsTimer.Start();

        g_hud.DrawString(10, -180, "Tess level  : %d", g_tessLevel);
        g_hud.DrawString(10, -160, "Primitives  : %d", numPrimsGenerated);
        g_hud.DrawString(10, -140, "Draw calls  : %d", numDrawCalls);
        g_hud.DrawString(10, -100, "GPU Compute : %.3f ms", g_gpuTime);
        g_hud.DrawString(10, -80,  "CPU Compute : %.3f ms", g_cpuTime);
        g_hud.DrawString(10, -60,  "GPU Draw    : %.3f ms", drawGpuTime);
        g_hud.DrawString(10, -40,  "CPU Draw    : %.3f ms", drawCpuTime);
        g_hud.DrawString(10, -20,  "FPS         : %3.1f", fps);

        g_hud.Flush();
    }

    glFinish();

    //checkGLErrors("display leave");
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
    } else if ((g_mbutton[0] && !g_mbutton[1] && g_mbutton[2]) or
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
uninitGL() {

    glDeleteQueries(2, g_queries);
    glDeleteVertexArrays(1, &g_vao);

    if (g_instances)
        delete g_instances;
    if (g_topology)
        delete g_topology;

#ifdef OPENSUBDIV_HAS_OPENCL
    uninitCL(g_clContext, g_clQueue);
#endif

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

static void
rebuildInstances() {

    delete g_instances;
    if (g_displayStyle == kVaryingInterleaved) {
        g_instances = g_topology->CreateInstances(
            g_numInstances,
            Osd::VertexBufferDescriptor(0, 3, 7),
            Osd::VertexBufferDescriptor(3, 4, 7),
            true);
    } else if (g_displayStyle == kVarying) {
        g_instances = g_topology->CreateInstances(
            g_numInstances,
            Osd::VertexBufferDescriptor(0, 3, 3),
            Osd::VertexBufferDescriptor(0, 4, 4),
            false);
    } else {
        g_instances = g_topology->CreateInstances(
            g_numInstances,
            Osd::VertexBufferDescriptor(0, 3, 3),
            Osd::VertexBufferDescriptor(0, 0, 0),
            false);
    }

    updateGeom();
    refine();
}

static void
rebuildOsdMesh() {

    static SimpleShape g_modelCube =
        SimpleShape(catmark_cube, "catmark_cube", kCatmark);
    //static SimpleShape g_modelBishop =
    // SimpleShape(catmark_bishop, "catmark_bishop", kCatmark);
    static SimpleShape g_modelPawn =
        SimpleShape(catmark_pawn, "catmark_pawn", kCatmark);
    // static SimpleShape g_modelRook =
    //     SimpleShape(catmark_rook, "catmark_rook", kCatmark);

    delete g_topology;
    g_topology = createOsdMesh(g_modelPawn.data, g_level);
    //g_topology = createOsdMesh(g_modelCube.data, g_level);

    rebuildInstances();
}

//------------------------------------------------------------------------------
static void
keyboard(GLFWwindow *, int key, int /* scancode */, int event, int /* mods */) {

    if (event == GLFW_RELEASE) return;
    if (g_hud.KeyDown(tolower(key))) return;

    if (key == 'G') {
        g_frame++;
        updateGeom();
        refine();
    }

    switch (key) {
        case 'Q': g_running = 0; break;
        case 'F': fitFrame(); break;
        case '+':
        case '=': g_tessLevel++; break;
        case '-': g_tessLevel = std::max(g_tessLevelMin, g_tessLevel-1); break;
        case '.': g_numInstances++; rebuildInstances(); break;
        case ',': g_numInstances = std::max(1, g_numInstances-1); rebuildInstances(); break;
        case GLFW_KEY_ESCAPE: g_hud.SetVisible(!g_hud.IsVisible()); break;
    }
}

//------------------------------------------------------------------------------

static void
callbackKernel(int k) {

    g_kernel = k;

#ifdef OPENSUBDIV_HAS_OPENCL
    if (g_kernel == kCL and g_clContext == NULL) {
        if (initCL(&g_clContext, &g_clQueue) == false) {
            printf("Error in initializing OpenCL\n");
            exit(1);
        }
    }
#endif

#ifdef OPENSUBDIV_HAS_CUDA
    if (g_kernel == kCUDA and g_cudaInitialized == false) {
        g_cudaInitialized = true;
        cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
    }
#endif

    rebuildOsdMesh();
}

static void
callbackLevel(int l) {

    g_level = l;
    rebuildOsdMesh();
}

static void
callbackSlider(float value, int /* data */) {

    g_numInstances = (int)value;
    rebuildInstances();
}

static void
callbackDisplayStyle(int b) {

    g_displayStyle = b;
    rebuildInstances();
}

static void
callbackAdaptive(bool checked, int /* a */) {

    if (Osd::GLDrawContext::SupportsAdaptiveTessellation()) {
        g_adaptive = checked;
        rebuildOsdMesh();
    }
}

static void
callbackCheckBox(bool checked, int button) {

    switch (button) {
    case kHUD_CB_FREEZE:
        g_freeze = checked;
        break;
    }
}

static void
initHUD() {

    int windowWidth = g_width, windowHeight = g_height,
        frameBufferWidth = g_width, frameBufferHeight = g_height;

    // window size might not match framebuffer size on a high DPI display
    glfwGetWindowSize(g_window, &windowWidth, &windowHeight);
    glfwGetFramebufferSize(g_window, &frameBufferWidth, &frameBufferHeight);

    g_hud.Init(windowWidth, windowHeight, frameBufferWidth, frameBufferHeight);

    g_hud.SetFrameBuffer(new GLFrameBuffer);

    int shading_pulldown = g_hud.AddPullDown("Shading (W)", 10, 10, 250, callbackDisplayStyle, 'w');
    g_hud.AddPullDownButton(shading_pulldown, "Wire", kWire, g_displayStyle==kWire);
    g_hud.AddPullDownButton(shading_pulldown, "Shaded", kShaded, g_displayStyle==kShaded);
    g_hud.AddPullDownButton(shading_pulldown, "Wire+Shaded", kWireShaded, g_displayStyle==kWireShaded);
    g_hud.AddPullDownButton(shading_pulldown, "Varying", kVarying, g_displayStyle==kVarying);
    g_hud.AddPullDownButton(shading_pulldown, "Varying(Interleaved)", kVaryingInterleaved, g_displayStyle==kVaryingInterleaved);

    g_hud.AddCheckBox("Freeze (spc)", g_freeze != 0,
                      10, 150, callbackCheckBox, kHUD_CB_FREEZE, ' ');

    int compute_pulldown = g_hud.AddPullDown("Compute (K)", 475, 10, 300, callbackKernel, 'k');
    g_hud.AddPullDownButton(compute_pulldown, "CPU", kCPU);
#ifdef OPENSUBDIV_HAS_OPENMP
    g_hud.AddPullDownButton(compute_pulldown, "OpenMP", kOPENMP);
#endif
#ifdef OPENSUBDIV_HAS_TBB
    g_hud.AddPullDownButton(compute_pulldown, "TBB", kTBB);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    g_hud.AddPullDownButton(compute_pulldown, "CUDA", kCUDA);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    if (HAS_CL_VERSION_1_1()) {
        g_hud.AddPullDownButton(compute_pulldown, "OpenCL", kCL);
    }
#endif
#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    g_hud.AddPullDownButton(compute_pulldown, "GLSL TransformFeedback", kGLSL);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    // Must also check at run time for OpenGL 4.3
    if (GLEW_VERSION_4_3) {
        g_hud.AddPullDownButton(compute_pulldown, "GLSL Compute", kGLSLCompute);
    }
#endif

    g_hud.AddSlider("Prim counts", 1, 100, 25,
                    -200, 20, 20, false, callbackSlider, 0);

    if (Osd::GLDrawContext::SupportsAdaptiveTessellation())
        g_hud.AddCheckBox("Adaptive (`)", g_adaptive!=0, 10, 190, callbackAdaptive, 0, '`');

    for (int i = 1; i < 11; ++i) {
        char level[16];
        sprintf(level, "Lv. %d", i);
        g_hud.AddRadioButton(3, level, i==2, 10, 210+i*20, callbackLevel, i, '0'+(i%10));
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

    glGenQueries(2, g_queries);

    glGenVertexArrays(1, &g_vao);
}

//------------------------------------------------------------------------------
static void
idle() {
    if (not g_freeze) {
        ++g_frame;
        updateGeom();
        refine();
    }
}

//------------------------------------------------------------------------------
static void
callbackError(Far::ErrorType err, const char *message) {
    printf("Error: %d\n", err);
    printf("%s", message);
}

//------------------------------------------------------------------------------
static void
callbackErrorGLFW(int error, const char* description) {
    fprintf(stderr, "GLFW Error (%d) : %s\n", error, description);
}
//------------------------------------------------------------------------------
static void
setGLCoreProfile() {

    #define glfwOpenWindowHint glfwWindowHint
    #define GLFW_OPENGL_VERSION_MAJOR GLFW_CONTEXT_VERSION_MAJOR
    #define GLFW_OPENGL_VERSION_MINOR GLFW_CONTEXT_VERSION_MINOR

    glfwOpenWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if not defined(__APPLE__)
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 4);
#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 3);
#else
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
#endif

#else
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3);
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
#endif
    glfwOpenWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
}

//------------------------------------------------------------------------------
int main(int argc, char ** argv) {

    std::string str;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-d")) {
            g_level = atoi(argv[++i]);
        }
    }
    Far::SetErrorCallback(callbackError);

    glfwSetErrorCallback(callbackErrorGLFW);
    if (not glfwInit()) {
        printf("Failed to initialize GLFW\n");
        return 1;
    }

    static const char windowTitle[] = "OpenSubdiv face partitioning example";

#define CORE_PROFILE
#ifdef CORE_PROFILE
    setGLCoreProfile();
#endif

    if (not (g_window=glfwCreateWindow(g_width, g_height, windowTitle, NULL, NULL))) {
        printf("Failed to open window.\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(g_window);

    // accommocate high DPI displays (e.g. mac retina displays)
    glfwGetFramebufferSize(g_window, &g_width, &g_height);
    glfwSetFramebufferSizeCallback(g_window, reshape);

    glfwSetKeyCallback(g_window, keyboard);
    glfwSetCursorPosCallback(g_window, motion);
    glfwSetMouseButtonCallback(g_window, mouse);
    glfwSetWindowCloseCallback(g_window, windowClose);

#if defined(OSD_USES_GLEW)
#ifdef CORE_PROFILE
    // this is the only way to initialize glew correctly under core profile context.
    glewExperimental = true;
#endif
    if (GLenum r = glewInit() != GLEW_OK) {
        printf("Failed to initialize glew. Error = %s\n", glewGetErrorString(r));
        exit(1);
    }
#ifdef CORE_PROFILE
    // clear GL errors which was generated during glewInit()
    glGetError();
#endif
#endif

    // activate feature adaptive tessellation if OSD supports it
    g_adaptive = Osd::GLDrawContext::SupportsAdaptiveTessellation();

    initGL();

    glfwSwapInterval(0);

    initHUD();
    rebuildOsdMesh();

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
