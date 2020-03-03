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

#include "../../regression/common/far_utils.h"
#include "../../regression/common/arg_utils.h"
#include "../common/viewerArgsUtils.h"
#include "../common/stopwatch.h"
#include "../common/simple_math.h"
#include "../common/glHud.h"
#include "../common/glControlMeshDisplay.h"
#include "../common/glUtils.h"

#include <opensubdiv/far/patchTableFactory.h>
#include <opensubdiv/far/ptexIndices.h>
#include <opensubdiv/far/stencilTableFactory.h>

#include <opensubdiv/osd/cpuGLVertexBuffer.h>
#include <opensubdiv/osd/cpuVertexBuffer.h>
#include <opensubdiv/osd/cpuEvaluator.h>

#if defined(OPENSUBDIV_HAS_OPENMP)
    #include <opensubdiv/osd/ompEvaluator.h>
#endif

#ifdef OPENSUBDIV_HAS_TBB
    #include <opensubdiv/osd/tbbEvaluator.h>
#endif

#ifdef OPENSUBDIV_HAS_CUDA
    #include <opensubdiv/osd/cudaVertexBuffer.h>
    #include <opensubdiv/osd/cudaGLVertexBuffer.h>
    #include <opensubdiv/osd/cudaEvaluator.h>
    #include "../common/cudaDeviceContext.h"

    CudaDeviceContext g_cudaDeviceContext;
#endif

#ifdef OPENSUBDIV_HAS_OPENCL
    #include <opensubdiv/osd/clVertexBuffer.h>
    #include <opensubdiv/osd/clGLVertexBuffer.h>
    #include <opensubdiv/osd/clEvaluator.h>
    #include "../common/clDeviceContext.h"

    CLDeviceContext g_clDeviceContext;
#endif

#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    #include <opensubdiv/osd/glXFBEvaluator.h>
    #include <opensubdiv/osd/glVertexBuffer.h>
#endif

#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    #include <opensubdiv/osd/glComputeEvaluator.h>
    #include <opensubdiv/osd/glVertexBuffer.h>
#endif

#include <opensubdiv/osd/mesh.h>

#include <cfloat>
#include <list>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>

using namespace OpenSubdiv;

enum KernelType { kCPU = 0,
                  kOPENMP,
                  kTBB,
                  kCUDA,
                  kCL,
                  kGLXFB,
                  kGLCompute };

enum HudCheckBox { kHUD_CB_DISPLAY_CONTROL_MESH_EDGES,
                   kHUD_CB_DISPLAY_CONTROL_MESH_VERTS,
                   kHUD_CB_ANIMATE_VERTICES,
                   kHUD_CB_FREEZE,
                   kHUD_CB_ADAPTIVE,
                   kHUD_CB_INF_SHARP_PATCH };

int g_kernel = kCPU,
    g_isolationLevel = 2; // max level of extraordinary feature isolation

int   g_running = 1,
      g_width = 1024,
      g_height = 1024,
      g_prev_x = 0,
      g_prev_y = 0,
      g_mbutton[3] = {0, 0, 0},
      g_frame=0,
      g_freeze=0,
      g_repeatCount=0;

bool g_adaptive=true,
     g_infSharpPatch=true;

float g_rotate[2] = {0, 0},
      g_dolly = 5,
      g_pan[2] = {0, 0},
      g_center[3] = {0, 0, 0},
      g_size = 0,
      g_moveScale = 0.0f;

bool  g_yup = false;

struct Transform {
    float ModelViewMatrix[16];
    float ProjectionMatrix[16];
    float ModelViewProjectionMatrix[16];
} g_transformData;


// performance
float g_evalTime = 0;
Stopwatch g_fpsTimer;

std::vector<float> g_orgPositions;
std::vector<float> g_positions;

int g_nsamples=2000,
    g_nsamplesDrawn=0;

GLuint g_stencilsVAO = 0;

GLhud g_hud;
GLControlMeshDisplay g_controlMeshDisplay;

//------------------------------------------------------------------------------

#include "init_shapes.h"

int g_currentShape = 0;

//------------------------------------------------------------------------------
Far::LimitStencilTable const * g_controlStencils;

class StencilOutputBase {
public:
    virtual ~StencilOutputBase() {}
    virtual void UpdateData(const float *src, int startVertex, int numVertices) = 0;
    virtual void EvalStencils() = 0;
    virtual GLuint BindSrcBuffer() = 0;
    virtual GLuint BindDstBuffer() = 0;
    virtual int GetNumStencils() const = 0;
};

template<typename SRC_BUFFER, typename DST_BUFFER,
         typename STENCIL_TABLE, typename EVALUATOR,
         typename DEVICE_CONTEXT=void>
class StencilOutput : public StencilOutputBase {
public:
    typedef OpenSubdiv::Osd::EvaluatorCacheT<EVALUATOR> EvaluatorCache;

    StencilOutput(Far::LimitStencilTable const *limitStencils,
                  int numSrcVerts,
                  EvaluatorCache *evaluatorCache = NULL,
                  DEVICE_CONTEXT *deviceContext = NULL)
        : _srcDesc(/*offset*/ 0, /*length*/ 3, /*stride*/ 3),
          _dstDesc(/*offset*/ 0, /*length*/ 3, /*stride*/ 9),
          _duDesc( /*offset*/ 3, /*length*/ 3, /*stride*/ 9),
          _dvDesc( /*offset*/ 6, /*length*/ 3, /*stride*/ 9),
          _deviceContext(deviceContext) {

        // src buffer  [ P(xyz) ]
        // dst buffer  [ P(xyz), du(xyz), dv(xyz) ]

        _numStencils = limitStencils->GetNumStencils();

        _srcData = SRC_BUFFER::Create(3, numSrcVerts, _deviceContext);
        _dstData = DST_BUFFER::Create(9, _numStencils, _deviceContext);

        _stencils =
            Osd::convertToCompatibleStencilTable<STENCIL_TABLE>(
                limitStencils, _deviceContext);
        _evaluatorCache = evaluatorCache;
    }
    ~StencilOutput() {
        delete _srcData;
        delete _dstData;
        delete _stencils;
    }
    virtual int GetNumStencils() const {
        return _numStencils;
    }
    virtual void UpdateData(const float *src, int startVertex, int numVertices) {
        _srcData->UpdateData(src, startVertex, numVertices, _deviceContext);
    };
    virtual void EvalStencils() {
        EVALUATOR const *evalInstance = OpenSubdiv::Osd::GetEvaluator<EVALUATOR>(
            _evaluatorCache, _srcDesc, _dstDesc, _duDesc, _dvDesc, _deviceContext);

        EVALUATOR::EvalStencils(_srcData, _srcDesc,
                                _dstData, _dstDesc,
                                _dstData, _duDesc,
                                _dstData, _dvDesc,
                                _stencils,
                                evalInstance,
                                _deviceContext);
    }
    virtual GLuint BindSrcBuffer() {
        return _srcData->BindVBO();
    }
    virtual GLuint BindDstBuffer() {
        return _dstData->BindVBO();
    }

private:
    SRC_BUFFER *_srcData;
    DST_BUFFER *_dstData;
    Osd::BufferDescriptor _srcDesc;
    Osd::BufferDescriptor _dstDesc;
    Osd::BufferDescriptor _duDesc;
    Osd::BufferDescriptor _dvDesc;

    STENCIL_TABLE const *_stencils;
    int _numStencils;

    EvaluatorCache *_evaluatorCache;
    DEVICE_CONTEXT *_deviceContext;
};

StencilOutputBase *g_stencilOutput = NULL;

//------------------------------------------------------------------------------
#define SCALE_TAN 0.02f
#define SCALE_NORM 0.02f

static void
updateGeom() {

    int nverts = (int)g_orgPositions.size() / 3;

    const float *p = &g_orgPositions[0];

    float r = sin(g_frame*0.001f) * g_moveScale;

    g_positions.resize(nverts*3);

    for (int i = 0; i < nverts; ++i) {
        //float move = 0.05f*cosf(p[0]*20+g_frame*0.01f);
        float ct = cos(p[2] * r);
        float st = sin(p[2] * r);
        g_positions[i*3+0] = p[0]*ct + p[1]*st;
        g_positions[i*3+1] = -p[0]*st + p[1]*ct;
        g_positions[i*3+2] = p[2];
        p+=3;
    }

    Stopwatch s;
    s.Start();

    // update control points
    g_stencilOutput->UpdateData(&g_positions[0], 0, nverts);

    // Update random points by applying point & tangent stencils
    g_stencilOutput->EvalStencils();


    s.Stop();
    g_evalTime = float(s.GetElapsed() * 1000.0f);
}

//------------------------------------------------------------------------------

static void
createMesh(ShapeDesc const & shapeDesc, int level) {

    typedef Far::LimitStencilTableFactory::LocationArray LocationArray;

    Shape const * shape = Shape::parseObj(shapeDesc);

    // create Far mesh (topology)
    Sdc::SchemeType sdctype = GetSdcType(*shape);
    Sdc::Options sdcoptions = GetSdcOptions(*shape);
    int regFaceSize = Sdc::SchemeTypeTraits::GetRegularFaceSize(sdctype);

    Far::TopologyRefiner * refiner =
        Far::TopologyRefinerFactory<Shape>::Create(*shape,
            Far::TopologyRefinerFactory<Shape>::Options(sdctype, sdcoptions));

    // save coarse topology (used for coarse mesh drawing)
    Far::TopologyLevel const & refBaseLevel = refiner->GetLevel(0);

    g_controlMeshDisplay.SetTopology(refBaseLevel);
    int nverts = refBaseLevel.GetNumVertices();

    // save rest pose
    g_orgPositions = shape->verts;

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

    if (!g_adaptive) {
        Far::TopologyRefiner::UniformOptions options(level);
        options.fullTopologyInLastLevel = true;
        refiner->RefineUniform(options);
    } else {
        Far::TopologyRefiner::AdaptiveOptions options(level);
        options.useSingleCreasePatch = false;
        options.useInfSharpPatch = g_infSharpPatch;
        refiner->RefineAdaptive(options);
    }

    Far::PtexIndices ptexIndices(*refiner);
    int nfaces = ptexIndices.GetNumFaces();

    float * u = new float[g_nsamples*nfaces], * uPtr = u,
          * v = new float[g_nsamples*nfaces], * vPtr = v;

    std::vector<LocationArray> locs(nfaces);

    srand( static_cast<int>(2147483647) ); // use a large Pell prime number
    for (int face=0; face<nfaces; ++face) {

        LocationArray & larray = locs[face];
        larray.ptexIdx = face;
        larray.numLocations = g_nsamples;
        larray.s = uPtr;
        larray.t = vPtr;

        for (int j=0; j<g_nsamples; ++j, ++uPtr, ++vPtr) {
            float u = (float)rand()/(float)RAND_MAX;
            float v = (float)rand()/(float)RAND_MAX;
            if ((regFaceSize==3) && (u+v >= 1.0f)) {
                // Keep locations within the triangular parametric domain
                u = 1.0f - u;
                v = 1.0f - v;
            }
            *uPtr = u;
            *vPtr = v;
        }
    }

    delete g_controlStencils;
    g_controlStencils = Far::LimitStencilTableFactory::Create(*refiner, locs);

    delete [] u;
    delete [] v;

    g_nsamplesDrawn = g_controlStencils->GetNumStencils();

    delete shape;
    delete refiner;

    delete g_stencilOutput;
    if (g_kernel == kCPU) {
        g_stencilOutput = new StencilOutput<Osd::CpuGLVertexBuffer,
                                            Osd::CpuGLVertexBuffer,
                                            Far::LimitStencilTable,
                                            Osd::CpuEvaluator>(
                                                g_controlStencils, nverts);
#ifdef OPENSUBDIV_HAS_OPENMP
    } else if (g_kernel == kOPENMP) {
        g_stencilOutput = new StencilOutput<Osd::CpuGLVertexBuffer,
                                            Osd::CpuGLVertexBuffer,
                                            Far::LimitStencilTable,
                                            Osd::OmpEvaluator>(
                                                g_controlStencils, nverts);
#endif
#ifdef OPENSUBDIV_HAS_TBB
    } else if (g_kernel == kTBB) {
        g_stencilOutput = new StencilOutput<Osd::CpuGLVertexBuffer,
                                            Osd::CpuGLVertexBuffer,
                                            Far::LimitStencilTable,
                                            Osd::TbbEvaluator>(
                                                g_controlStencils, nverts);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    } else if (g_kernel == kCUDA) {
        g_stencilOutput = new StencilOutput<Osd::CudaGLVertexBuffer,
                                            Osd::CudaGLVertexBuffer,
                                            Osd::CudaStencilTable,
                                            Osd::CudaEvaluator>(
                                                g_controlStencils, nverts);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    } else if (g_kernel == kCL) {
        static Osd::EvaluatorCacheT<Osd::CLEvaluator> clEvaluatorCache;
        g_stencilOutput = new StencilOutput<Osd::CLGLVertexBuffer,
                                            Osd::CLGLVertexBuffer,
                                            Osd::CLStencilTable,
                                            Osd::CLEvaluator,
                                            CLDeviceContext>(
                                                g_controlStencils, nverts,
                                                &clEvaluatorCache,
                                                &g_clDeviceContext);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    } else if (g_kernel == kGLXFB) {
        static Osd::EvaluatorCacheT<Osd::GLXFBEvaluator> glXFBEvaluatorCache;
        g_stencilOutput = new StencilOutput<Osd::GLVertexBuffer,
                                            Osd::GLVertexBuffer,
                                            Osd::GLStencilTableTBO,
                                            Osd::GLXFBEvaluator>(
                                                g_controlStencils, nverts,
                                                 &glXFBEvaluatorCache);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    } else if (g_kernel == kGLCompute) {
        static Osd::EvaluatorCacheT<Osd::GLComputeEvaluator> glComptueEvaluatorCache;
        g_stencilOutput = new StencilOutput<Osd::GLVertexBuffer,
                                            Osd::GLVertexBuffer,
                                            Osd::GLStencilTableSSBO,
                                            Osd::GLComputeEvaluator>(
                                                g_controlStencils, nverts,
                                                    &glComptueEvaluatorCache);
#endif
    }

    updateGeom();
}

//------------------------------------------------------------------------------
class GLSLProgram {
public:
    GLSLProgram() : _program(0), _vtxSrc(0), _frgSrc(0) { }

    struct Attribute {
        std::string name;
        GLuint location;
        GLuint size;
    };

    void SetVertexShaderSource( char const * src ) {
        _vtxSrc = src;
    }

    void SetGeometryShaderSource( char const * src) {
        _geomSrc = src;
    }

    void SetFragShaderSource( char const * src ) {
        _frgSrc = src;
    }

    void AddAttribute( char const * attr, int size ) {
        Attribute a;
        a.name = attr;
        a.size = size;
        _attrs.push_back(a);
    }

    void EnableVertexAttributes( ) {

        GLvoid * offset = 0;
        for (AttrList::iterator i=_attrs.begin(); i!=_attrs.end(); ++i) {

            glEnableVertexAttribArray( i->location );

            glVertexAttribPointer( i->location, i->size,
                GL_FLOAT, GL_FALSE, sizeof(GLfloat) * _attrStride, (GLvoid*)offset);

            offset = (GLubyte*)offset + sizeof(GLfloat) * i->size;
        }
    }
    GLuint GetUniformScale() const {
        return _uniformScale;
    }
    GLuint GetUniformProjectionMatrix() const {
        return _uniformProjectionMatrix;
    }
    GLuint GetUniformModelViewMatrix() const {
        return _uniformModelViewMatrix;
    }
    GLuint GetUniformModelViewProjectionMatrix() const {
        return _uniformModelViewProjectionMatrix;
    }

    void Use( ) {

        if (! _program) {
            assert( _vtxSrc && _frgSrc );

            _program = glCreateProgram();

            GLuint vertexShader =
                GLUtils::CompileShader(GL_VERTEX_SHADER, _vtxSrc);
            GLuint fragmentShader =
                GLUtils::CompileShader(GL_FRAGMENT_SHADER, _frgSrc);

            glAttachShader(_program, vertexShader);
            glAttachShader(_program, fragmentShader);

            GLuint geomShader = 0;
            if (_geomSrc) {
                geomShader = GLUtils::CompileShader(GL_GEOMETRY_SHADER, _geomSrc);
                glAttachShader(_program, geomShader);
            }

            _attrStride=0;
            int count=0;
            for (AttrList::iterator i=_attrs.begin(); i!=_attrs.end(); ++i, ++count) {
                glBindAttribLocation(_program, count, i->name.c_str());
                _attrStride += i->size;
            }

            glBindFragDataLocation(_program, 0, "color");

            glLinkProgram(_program);

            GLint status;
            glGetProgramiv(_program, GL_LINK_STATUS, &status);
            if (status == GL_FALSE) {
                GLint infoLogLength;
                glGetProgramiv(_program, GL_INFO_LOG_LENGTH, &infoLogLength);
                char *infoLog = new char[infoLogLength];
                glGetProgramInfoLog(_program, infoLogLength, NULL, infoLog);
                printf("%s\n", infoLog);
                delete[] infoLog;
                exit(1);
            }

            _uniformScale =
                glGetUniformLocation(_program, "scale");
            _uniformModelViewMatrix =
                glGetUniformLocation(_program, "ModelViewMatrix");
            _uniformProjectionMatrix =
                glGetUniformLocation(_program, "ProjectionMatrix");
            _uniformModelViewProjectionMatrix =
                glGetUniformLocation(_program, "ModelViewProjectionMatrix");

            for (AttrList::iterator i=_attrs.begin(); i!=_attrs.end(); ++i) {
                i->location = glGetAttribLocation(_program, i->name.c_str());
            }
        }

        glUseProgram(_program);
    }

private:

    GLuint _program;
    GLuint _uniformScale;
    GLuint _uniformModelViewMatrix;
    GLuint _uniformProjectionMatrix;
    GLuint _uniformModelViewProjectionMatrix;

    char const * _vtxSrc,
               * _geomSrc,
               * _frgSrc;

    typedef std::list<Attribute> AttrList;
    AttrList _attrs;
    int _attrStride;

};

GLSLProgram g_samplesProgram;


//------------------------------------------------------------------------------
static bool
linkDefaultPrograms() {

#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
    #define GLSL_VERSION_DEFINE "#version 400\n"
#else
    #define GLSL_VERSION_DEFINE "#version 150\n"
#endif
    {   // setup samples program
        //
        // this shader takes position, uTangent and vTangent for each point
        // then generates 3 lines in the geometry shader.
        //
        static const char *vsSrc =
            GLSL_VERSION_DEFINE
            "in vec3 position;\n"
            "in vec3 uTangent;\n"
            "in vec3 vTangent;\n"
            "out vec3 p;\n"
            "out vec3 ut;\n"
            "out vec3 vt;\n"
            "uniform mat4 ModelViewMatrix;\n"
            "void main() {\n"
            "  p =  (ModelViewMatrix * vec4(position, 1)).xyz;\n"
            "  ut = (ModelViewMatrix * vec4(uTangent, 0)).xyz;\n"
            "  vt = (ModelViewMatrix * vec4(vTangent, 0)).xyz;\n"
            "}\n";

        static const char *gsSrc =
            GLSL_VERSION_DEFINE
            "layout(points) in;\n"
            "layout(line_strip, max_vertices = 6) out;\n"
            "in vec3 p[];\n"
            "in vec3 ut[];\n"
            "in vec3 vt[];\n"
            "out vec4 c;\n"
            "uniform mat4 ProjectionMatrix;\n"
            "uniform float scale;\n"
            "void main() {\n"
            "  vec3 pos = p[0]; \n"
            "  c = vec4(1, 0, 0, 1);\n"
            "  gl_Position = ProjectionMatrix * vec4(pos, 1);\n"
            "  EmitVertex();\n"
            "  \n"
            "  pos = p[0] + ut[0] * scale; \n"
            "  gl_Position = ProjectionMatrix * vec4(pos, 1);\n"
            "  EmitVertex();\n"
            "  EndPrimitive();\n"
            "  \n"
            "   pos = p[0]; \n"
            "  c = vec4(0, 1, 0, 1);\n"
            "  gl_Position = ProjectionMatrix * vec4(pos, 1);\n"
            "  EmitVertex();\n"
            "  \n"
            "  pos = p[0] + vt[0] * scale; \n"
            "  gl_Position = ProjectionMatrix * vec4(pos, 1);\n"
            "  EmitVertex();\n"
            "  EndPrimitive();\n"
            "  \n"
            "  pos = p[0]; \n"
            "  c = vec4(0, 0, 1, 1);\n"
            "  gl_Position = ProjectionMatrix * vec4(pos, 1);\n"
            "  EmitVertex();\n"
            "  \n"
            "  pos = p[0] + cross(ut[0], vt[0]) * scale; \n"
            "  gl_Position = ProjectionMatrix * vec4(pos, 1);\n"
            "  EmitVertex();\n"
            "  EndPrimitive();\n"
            "  \n"
            "}\n";

        static const char *fsSrc =
            GLSL_VERSION_DEFINE
            "in vec4 c;\n"
            "out vec4 color;\n"
            "void main() {\n"
            "   color = c;\n"
            "}\n";

        g_samplesProgram.SetVertexShaderSource(vsSrc);
        g_samplesProgram.SetGeometryShaderSource(gsSrc);
        g_samplesProgram.SetFragShaderSource(fsSrc);

        g_samplesProgram.AddAttribute( "position",3 );
        g_samplesProgram.AddAttribute( "uTangent",3 );
        g_samplesProgram.AddAttribute( "vTangent",3 );
    }

    return true;
}

//------------------------------------------------------------------------------
static void
drawStencils() {

    g_samplesProgram.Use( );

    const float scale = 0.02f;

    glUniform1f(g_samplesProgram.GetUniformScale(), scale);
    glUniformMatrix4fv(g_samplesProgram.GetUniformModelViewMatrix(),
                       1, GL_FALSE, g_transformData.ModelViewMatrix);
    glUniformMatrix4fv(g_samplesProgram.GetUniformProjectionMatrix(),
                       1, GL_FALSE, g_transformData.ProjectionMatrix);

    glBindVertexArray(g_stencilsVAO);

    glBindBuffer(GL_ARRAY_BUFFER, g_stencilOutput->BindDstBuffer());

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat)*9, 0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat)*9, (void*)(sizeof(GLfloat)*3));
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat)*9, (void*)(sizeof(GLfloat)*6));

    glDrawArrays(GL_POINTS, 0, g_stencilOutput->GetNumStencils());

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);

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

    drawStencils();

    // draw the control mesh
    g_controlMeshDisplay.Draw(g_stencilOutput->BindSrcBuffer(), 3*sizeof(float),
                              g_transformData.ModelViewProjectionMatrix);

    s.Stop();
    float drawCpuTime = float(s.GetElapsed() * 1000.0f);
    s.Start();
    glFinish();
    s.Stop();
    float drawGpuTime = float(s.GetElapsed() * 1000.0f);

    if (g_hud.IsVisible()) {
        g_fpsTimer.Stop();
        double fps = 1.0/g_fpsTimer.GetElapsed();
        g_fpsTimer.Start();

        g_hud.DrawString(10, -100,  "# stencils   : %d", g_nsamplesDrawn);
        g_hud.DrawString(10, -80,  "EvalStencils : %.3f ms", g_evalTime);
        g_hud.DrawString(10, -60,  "GPU Draw     : %.3f ms", drawGpuTime);
        g_hud.DrawString(10, -40,  "CPU Draw     : %.3f ms", drawCpuTime);
        g_hud.DrawString(10, -20,  "FPS          : %3.1f", fps);

        g_hud.Flush();
    }
    glFinish();

    //checkGLErrors("display leave");
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
rebuildMesh() {

    createMesh( g_defaultShapes[g_currentShape], g_isolationLevel );
}


//------------------------------------------------------------------------------
static void
setSamples(bool add) {

    g_nsamples += add ? 1000 : -1000;

    g_nsamples = std::max(1000, g_nsamples);

    rebuildMesh();
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

        case GLFW_KEY_ESCAPE: g_hud.SetVisible(!g_hud.IsVisible()); break;
    }
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

    rebuildMesh();
}

static void
callbackLevel(int l) {

    g_isolationLevel = l;

    rebuildMesh();
}

//------------------------------------------------------------------------------
static void
callbackModel(int m) {

    if (m < 0)
        m = 0;

    if (m >= (int)g_defaultShapes.size())
        m = (int)g_defaultShapes.size() - 1;

    g_currentShape = m;

    rebuildMesh();
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
    case kHUD_CB_FREEZE:
        g_freeze = checked;
        break;
    case kHUD_CB_ADAPTIVE:
        g_adaptive = checked;
        rebuildMesh();
        break;
    case kHUD_CB_INF_SHARP_PATCH:
        g_infSharpPatch = checked;
        rebuildMesh();
        break;
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
    g_hud.AddCheckBox("Freeze (spc)", g_freeze != 0,
                      10, 70, callbackCheckBox, kHUD_CB_FREEZE, ' ');

    g_hud.AddCheckBox("Adaptive (`)", g_adaptive != 0,
                      10, 190, callbackCheckBox, kHUD_CB_ADAPTIVE, '`');
    g_hud.AddCheckBox("Inf Sharp Patch  (I)", g_infSharpPatch != 0,
                      10, 210, callbackCheckBox, kHUD_CB_INF_SHARP_PATCH, 'i');

    int compute_pulldown = g_hud.AddPullDown("Compute (K)", 250, 10, 300, callbackKernel, 'k');
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

    for (int i = 1; i < 11; ++i) {
        char level[16];
        sprintf(level, "Lv. %d", i);
        g_hud.AddRadioButton(3, level, i==g_isolationLevel, 10, 250+i*20, callbackLevel, i, '0'+(i%10));
    }

    int pulldown_handle = g_hud.AddPullDown("Shape (N)", -300, 10, 300, callbackModel, 'n');
    for (int i = 0; i < (int)g_defaultShapes.size(); ++i) {
        g_hud.AddPullDownButton(pulldown_handle, g_defaultShapes[i].name.c_str(),i);
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

    glGenVertexArrays(1, &g_stencilsVAO);
}

//------------------------------------------------------------------------------
static void
uninitGL() {
    glDeleteVertexArrays(1, &g_stencilsVAO);
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
    g_isolationLevel = args.GetLevel();
    g_repeatCount = args.GetRepeatCount();

    ViewerArgsUtils::PopulateShapes(args, &g_defaultShapes);

    initShapes();

    glfwSetErrorCallback(callbackErrorGLFW);
    if (! glfwInit()) {
        printf("Failed to initialize GLFW\n");
        return 1;
    }

    static const char windowTitle[] = "OpenSubdiv glStencilViewer " OPENSUBDIV_VERSION_STRING;

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
    linkDefaultPrograms();

    glfwSwapInterval(0);

    initHUD();
    callbackModel(g_currentShape);

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
