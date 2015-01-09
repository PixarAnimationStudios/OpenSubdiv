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

#include <common/vtr_utils.h>
#include "../common/stopwatch.h"
#include "../common/simple_math.h"
#include "../common/gl_common.h"
#include "../common/gl_hud.h"

#include <far/patchTablesFactory.h>
#include <far/stencilTablesFactory.h>

#include <osd/cpuGLVertexBuffer.h>
#include <osd/cpuVertexBuffer.h>
#include <osd/cpuEvalStencilsContext.h>
#include <osd/cpuEvalStencilsController.h>

#include <cfloat>
#include <list>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>

#ifdef OPENSUBDIV_HAS_OPENMP
    #include <omp.h>
#endif

using namespace OpenSubdiv;

enum KernelType { kCPU    = 0,
                  kOPENMP = 1,
                  kTBB    = 2 };


int g_kernel = kCPU,
    g_isolationLevel = 5; // max level of extraordinary feature isolation

int   g_running = 1,
      g_width = 1024,
      g_height = 1024,
      g_fullscreen = 0,
      g_drawCageEdges = 1,
      g_drawCageVertices = 1,
      g_prev_x = 0,
      g_prev_y = 0,
      g_mbutton[3] = {0, 0, 0},
      g_frame=0,
      g_freeze=0,
      g_repeatCount;

bool g_bilinear=false;

float g_rotate[2] = {0, 0},
      g_dolly = 5,
      g_pan[2] = {0, 0},
      g_center[3] = {0, 0, 0},
      g_size = 0,
      g_moveScale = 0.0f;

struct Transform {
    float ModelViewMatrix[16];
    float ProjectionMatrix[16];
    float ModelViewProjectionMatrix[16];
} g_transformData;


// performance
float g_evalTime = 0;
Stopwatch g_fpsTimer;

std::vector<float> g_orgPositions;

std::vector<int>   g_coarseEdges;
std::vector<float> g_coarseEdgeSharpness;
std::vector<float> g_coarseVertexSharpness;

int g_nsamples=2000,
    g_nsamplesDrawn=0;

GLuint g_cageEdgeVAO = 0,
       g_cageEdgeVBO = 0,
       g_cageVertexVAO = 0,
       g_cageVertexVBO = 0,
       g_stencilsVAO = 0;

GLhud g_hud;


//------------------------------------------------------------------------------

#include "init_shapes.h"

int g_currentShape = 0;

//------------------------------------------------------------------------------
Far::LimitStencilTables const * g_controlStencils;

// Control vertex positions (P(xyz))
Osd::CpuVertexBuffer * g_controlValues=0;

// Display VBO (collects outputs of updated stencils)
Osd::CpuGLVertexBuffer * g_stencilValues=0;

// Display 3 lines for each stencil sample (utan, vtan, normal)
// 18 elements : [ P (xyz), P+dPdu (xyz),
//                 P (xyz), P+dPdv (xyz),
//                 P (xyz), P+N (xyz)    ]
Osd::VertexBufferDescriptor g_controlDesc( /*offset*/ 0, /*legnth*/ 3, /*stride*/ 3 ),
                            g_outputDataDesc( /*offset*/ 0, /*legnth*/ 3, /*stride*/ 18 ),
                            g_outputDuDesc( /*offset*/ 3, /*legnth*/ 3, /*stride*/ 18 ),
                            g_outputDvDesc( /*offset*/ 9, /*legnth*/ 3, /*stride*/ 18 );

Osd::CpuEvalStencilsContext * g_evalCtx=0;

Osd::CpuEvalStencilsController g_evalCpuCtrl;

#if defined(OPENSUBDIV_HAS_OPENMP)
    #include <osd/ompEvalStencilsController.h>
    Osd::OmpEvalStencilsController g_evalOmpCtrl;
#endif

#ifdef OPENSUBDIV_HAS_TBB
    #include <osd/tbbEvalStencilsController.h>
    Osd::TbbEvalStencilsController g_evalTbbCtrl;
#endif


//------------------------------------------------------------------------------
#define SCALE_TAN 0.02f
#define SCALE_NORM 0.02f

static void
updateGeom() {

    int nverts = (int)g_orgPositions.size() / 3;

    const float *p = &g_orgPositions[0];

    float r = sin(g_frame*0.001f) * g_moveScale;

    float * positions = g_controlValues->BindCpuBuffer();

    for (int i = 0; i < nverts; ++i) {
        //float move = 0.05f*cosf(p[0]*20+g_frame*0.01f);
        float ct = cos(p[2] * r);
        float st = sin(p[2] * r);
        positions[i*3+0] = p[0]*ct + p[1]*st;
        positions[i*3+1] = -p[0]*st + p[1]*ct;
        positions[i*3+2] = p[2];
        p+=3;
    }

    Stopwatch s;
    s.Start();

    float * ptr = g_stencilValues->BindCpuBuffer();
    memset(ptr, 0, g_controlStencils->GetNumStencils() * 18 * sizeof(float));

    // Uppdate random points by applying point & tangent stencils
    switch (g_kernel) {
        case kCPU: {
            g_evalCpuCtrl.UpdateValues<Osd::CpuVertexBuffer,Osd::CpuGLVertexBuffer>(
                g_evalCtx,
                g_controlDesc, g_controlValues,
                g_outputDataDesc, g_stencilValues );

            g_evalCpuCtrl.UpdateDerivs<Osd::CpuVertexBuffer,Osd::CpuGLVertexBuffer>(
                g_evalCtx,
                g_controlDesc, g_controlValues,
                g_outputDuDesc, g_stencilValues,
                g_outputDvDesc, g_stencilValues );
        } break;

#if defined(OPENSUBDIV_HAS_OPENMP)
        case kOPENMP: {
            g_evalOmpCtrl.UpdateValues<Osd::CpuVertexBuffer,Osd::CpuGLVertexBuffer>(
                g_evalCtx,
                g_controlDesc, g_controlValues,
                g_outputDataDesc, g_stencilValues );

            g_evalOmpCtrl.UpdateDerivs<Osd::CpuVertexBuffer,Osd::CpuGLVertexBuffer>(
                g_evalCtx,
                g_controlDesc, g_controlValues,
                g_outputDuDesc, g_stencilValues,
                g_outputDvDesc, g_stencilValues );
        } break;
#endif

#if defined(OPENSUBDIV_HAS_TBB)
        case kTBB: {
            g_evalTbbCtrl.UpdateValues<Osd::CpuVertexBuffer,Osd::CpuGLVertexBuffer>(
                g_evalCtx,
                g_controlDesc, g_controlValues,
                g_outputDataDesc, g_stencilValues );

            g_evalTbbCtrl.UpdateDerivs<Osd::CpuVertexBuffer,Osd::CpuGLVertexBuffer>(
                g_evalCtx,
                g_controlDesc, g_controlValues,
                g_outputDuDesc, g_stencilValues,
                g_outputDvDesc, g_stencilValues );
        } break;
#endif
        default:
            return;
     }

    s.Stop();
    g_evalTime = float(s.GetElapsed() * 1000.0f);

    assert(g_controlStencils);

    for (int i=0; i < g_controlStencils->GetNumStencils(); ++i, ptr+=18) {

        float * p      = ptr,
              * utan   = ptr + 3,
              * vtan   = ptr + 9,
              * normal = ptr + 15;

        // copy P as starting point for each line
        memcpy( ptr +  6, p, 3*sizeof(float) );
        memcpy( ptr + 12, p, 3*sizeof(float) );

        normalize( utan );
        normalize( vtan );
        cross( normal, utan, vtan );

        normalize(normal);

        // compute end point for each line (P + vec * scale)
        for (int j=0; j<3; ++j) {
            utan[j]= p[j] + utan[j]*SCALE_TAN;
            vtan[j]= p[j] + vtan[j]*SCALE_TAN;
            normal[j]= p[j] + normal[j]*SCALE_NORM;
        }
    }
}

//------------------------------------------------------------------------------

static void
createMesh(ShapeDesc const & shapeDesc, int level) {

    typedef Far::ConstIndexArray IndexArray;
    typedef Far::LimitStencilTablesFactory::LocationArray LocationArray;

    Shape const * shape = Shape::parseObj(shapeDesc.data.c_str(), shapeDesc.scheme);

    // create Vtr mesh (topology)
    OpenSubdiv::Sdc::SchemeType sdctype = GetSdcType(*shape);
    OpenSubdiv::Sdc::Options sdcoptions = GetSdcOptions(*shape);

    OpenSubdiv::Far::TopologyRefiner * refiner =
        OpenSubdiv::Far::TopologyRefinerFactory<Shape>::Create(*shape,
            OpenSubdiv::Far::TopologyRefinerFactory<Shape>::Options(sdctype, sdcoptions));

    // save coarse topology (used for coarse mesh drawing)
    int nedges = refiner->GetNumEdges(0),
        nverts = refiner->GetNumVertices(0);

    g_coarseEdges.resize(nedges*2);
    g_coarseEdgeSharpness.resize(nedges);
    g_coarseVertexSharpness.resize(nverts);

    for(int i=0; i<nedges; ++i) {
        IndexArray verts = refiner->GetEdgeVertices(0, i);
        g_coarseEdges[i*2  ]=verts[0];
        g_coarseEdges[i*2+1]=verts[1];
        g_coarseEdgeSharpness[i]=refiner->GetEdgeSharpness(0, i);
    }

    for(int i=0; i<nverts; ++i) {
        g_coarseVertexSharpness[i]=refiner->GetVertexSharpness(0, i);
    }

    g_orgPositions=shape->verts;

    if (g_bilinear) {
        Far::TopologyRefiner::UniformOptions options(level);
        options.fullTopologyInLastLevel = true;
        refiner->RefineUniform(options);
    } else {
        Far::TopologyRefiner::AdaptiveOptions options(level);
        options.fullTopologyInLastLevel = false;
        options.useSingleCreasePatch = false;
        refiner->RefineAdaptive(options);
    }

    int nfaces = refiner->GetNumPtexFaces();

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
            *uPtr = (float)rand()/(float)RAND_MAX;
            *vPtr = (float)rand()/(float)RAND_MAX;
        }
    }

    delete g_controlStencils;
    g_controlStencils = Far::LimitStencilTablesFactory::Create(*refiner, locs);

    delete [] u;
    delete [] v;

    g_nsamplesDrawn = g_controlStencils->GetNumStencils();

    // Create control vertex buffer (layout: [ P(xyz) ] )
    delete g_controlValues;
    g_controlValues = Osd::CpuVertexBuffer::Create(3, nverts);

    // Create eval context & data buffers
    delete g_evalCtx;
    g_evalCtx = Osd::CpuEvalStencilsContext::Create(g_controlStencils);

    delete g_stencilValues;
    g_stencilValues = Osd::CpuGLVertexBuffer::Create(3, g_controlStencils->GetNumStencils() * 6 );

    delete shape;
    delete refiner;

    updateGeom();

    // Bind g_stencilValues as GL_LINES VAO
    glBindVertexArray(g_stencilsVAO);

    glBindBuffer(GL_ARRAY_BUFFER, g_stencilValues->BindVBO());

    glBindVertexArray(0);

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

        long int offset = 0;
        for (AttrList::iterator i=_attrs.begin(); i!=_attrs.end(); ++i) {

            glEnableVertexAttribArray( i->location );

            glVertexAttribPointer( i->location, i->size,
                GL_FLOAT, GL_FALSE, sizeof(GLfloat) * _attrStride, (void*)offset);

            offset += sizeof(GLfloat) * i->size;
        }
    }

    GLuint GetUniformModelViewProjectionMatrix() const {
        return _uniformModelViewProjectionMatrix;
    }

    void Use( ) {

        if (not _program) {
            assert( _vtxSrc and _frgSrc );

            _program = glCreateProgram();

            GLuint vertexShader = compileShader(GL_VERTEX_SHADER, _vtxSrc);
            GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, _frgSrc);

            glAttachShader(_program, vertexShader);
            glAttachShader(_program, fragmentShader);

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
    GLuint _uniformModelViewProjectionMatrix;

    char const * _vtxSrc,
               * _frgSrc;

    typedef std::list<Attribute> AttrList;
    AttrList _attrs;
    int _attrStride;

};

GLSLProgram g_cageProgram,
            g_samplesProgram;


//------------------------------------------------------------------------------
static bool
linkDefaultPrograms() {

#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
    #define GLSL_VERSION_DEFINE "#version 400\n"
#else
    #define GLSL_VERSION_DEFINE "#version 150\n"
#endif

    {   // setup control cage program
        static const char *vsSrc =
            GLSL_VERSION_DEFINE
            "in vec3 position;\n"
            "in vec3 color;\n"
            "out vec4 fragColor;\n"
            "uniform mat4 ModelViewProjectionMatrix;\n"
            "void main() {\n"
            "  fragColor = vec4(color, 1);\n"
            "  gl_Position = ModelViewProjectionMatrix * "
            "                  vec4(position, 1);\n"
            "}\n";

        static const char *fsSrc =
            GLSL_VERSION_DEFINE
            "in vec4 fragColor;\n"
            "out vec4 color;\n"
            "void main() {\n"
            "  color = fragColor;\n"
            "}\n";

        g_cageProgram.SetVertexShaderSource(vsSrc);
        g_cageProgram.SetFragShaderSource(fsSrc);

        g_cageProgram.AddAttribute( "position",3 );
        g_cageProgram.AddAttribute( "color",3 );
    }

    {   // setup samples program
        static const char *vsSrc =
            GLSL_VERSION_DEFINE
            "in vec3 position;\n"
            "uniform mat4 ModelViewProjectionMatrix;\n"
            "void main() {\n"
            "  gl_Position = ModelViewProjectionMatrix * "
            "                  vec4(position, 1);\n"
            "}\n";

        static const char *fsSrc =
            GLSL_VERSION_DEFINE
            "out vec4 color;\n"
            "const vec4 colors[3] = vec4[3]( vec4(1.0,0.0,0.0,1.0),    \n"
            "                                vec4(0.0,1.0,0.0,1.0),    \n"
            "                                vec4(0.0,0.0,1.0,1.0)  ); \n"
            "void main() {\n"
            "   color = colors[gl_PrimitiveID % 3];\n"
            "}\n";

        g_samplesProgram.SetVertexShaderSource(vsSrc);
        g_samplesProgram.SetFragShaderSource(fsSrc);

        g_samplesProgram.AddAttribute( "position",3 );
    }

    return true;
}
//------------------------------------------------------------------------------
static inline void
setSharpnessColor(float s, float *r, float *g, float *b) {

    //  0.0       2.0       4.0
    // green --- yellow --- red
    *r = std::min(1.0f, s * 0.5f);
    *g = std::min(1.0f, 2.0f - s*0.5f);
    *b = 0;
}

//------------------------------------------------------------------------------
static void
drawCageEdges() {

    g_cageProgram.Use( );

    glUniformMatrix4fv(g_cageProgram.GetUniformModelViewProjectionMatrix(),
                       1, GL_FALSE, g_transformData.ModelViewProjectionMatrix);

    std::vector<float> vbo;
    vbo.reserve(g_coarseEdges.size() * 6);

    float * positions = g_controlValues->BindCpuBuffer();

    float r, g, b;
    for (int i = 0; i < (int)g_coarseEdges.size(); i+=2) {
        setSharpnessColor(g_coarseEdgeSharpness[i/2], &r, &g, &b);
        for (int j = 0; j < 2; ++j) {
            vbo.push_back(positions[g_coarseEdges[i+j]*3]);
            vbo.push_back(positions[g_coarseEdges[i+j]*3+1]);
            vbo.push_back(positions[g_coarseEdges[i+j]*3+2]);
            vbo.push_back(r);
            vbo.push_back(g);
            vbo.push_back(b);
        }
    }

    glBindVertexArray(g_cageEdgeVAO);

    glBindBuffer(GL_ARRAY_BUFFER, g_cageEdgeVBO);
    glBufferData(GL_ARRAY_BUFFER, (int)vbo.size() * sizeof(float), &vbo[0],
                 GL_STATIC_DRAW);

    g_cageProgram.EnableVertexAttributes();

    glDrawArrays(GL_LINES, 0, (int)g_coarseEdges.size());

    glBindVertexArray(0);
    glUseProgram(0);
}

//------------------------------------------------------------------------------
static void
drawCageVertices() {

    g_cageProgram.Use( );

    glUniformMatrix4fv(g_cageProgram.GetUniformModelViewProjectionMatrix(),
                       1, GL_FALSE, g_transformData.ModelViewProjectionMatrix);

    int numPoints = g_controlValues->GetNumVertices();
    std::vector<float> vbo;
    vbo.reserve(numPoints*6);

    float * positions = g_controlValues->BindCpuBuffer();

    float r, g, b;
    for (int i = 0; i < numPoints; ++i) {

        setSharpnessColor(g_coarseVertexSharpness[i], &r, &g, &b);

        vbo.push_back(positions[i*3+0]);
        vbo.push_back(positions[i*3+1]);
        vbo.push_back(positions[i*3+2]);
        vbo.push_back(r);
        vbo.push_back(g);
        vbo.push_back(b);
    }

    glBindVertexArray(g_cageVertexVAO);

    glBindBuffer(GL_ARRAY_BUFFER, g_cageVertexVBO);
    glBufferData(GL_ARRAY_BUFFER, (int)vbo.size() * sizeof(float), &vbo[0],
                 GL_STATIC_DRAW);

    g_cageProgram.EnableVertexAttributes();

    glPointSize(10.0f);
    glDrawArrays(GL_POINTS, 0, numPoints);
    glPointSize(1.0f);

    glBindVertexArray(0);
    glUseProgram(0);
}

//------------------------------------------------------------------------------
static void
drawStencils() {

    g_samplesProgram.Use( );

    glUniformMatrix4fv(g_cageProgram.GetUniformModelViewProjectionMatrix(),
                       1, GL_FALSE, g_transformData.ModelViewProjectionMatrix);

    glBindVertexArray(g_stencilsVAO);

    int numEdges = g_controlStencils->GetNumStencils() * 3;

    g_samplesProgram.EnableVertexAttributes();

    glBindBuffer(GL_ARRAY_BUFFER, g_stencilValues->BindVBO());

    g_samplesProgram.EnableVertexAttributes();

    glDrawArrays(GL_LINES, 0, numEdges*2);

    glBindVertexArray(0);
    glUseProgram(0);
}

//------------------------------------------------------------------------------
static void
display() {

    g_hud.GetFrameBuffer()->Bind();

    Stopwatch s;
    s.Start();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glViewport(0, 0, g_width, g_height);

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

    if (g_drawCageEdges)
        drawCageEdges();

    if (g_drawCageVertices)
        drawCageVertices();

    drawStencils();
    s.Stop();
    float drawCpuTime = float(s.GetElapsed() * 1000.0f);
    s.Start();
    glFinish();
    s.Stop();
    float drawGpuTime = float(s.GetElapsed() * 1000.0f);

    g_hud.GetFrameBuffer()->ApplyImageShader();

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

    if (not g_freeze)
        g_frame++;

    updateGeom();

    if (g_repeatCount != 0 and g_frame >= g_repeatCount)
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
keyboard(GLFWwindow *, int key, int /* scancode */, int event, int /* mods */) {

    if (event == GLFW_RELEASE) return;
    if (g_hud.KeyDown(tolower(key))) return;

    switch (key) {
        case 'Q': g_running = 0; break;

        case '=': setSamples(true); break;

        case '-': setSamples(false); break;

        case GLFW_KEY_ESCAPE: g_hud.SetVisible(!g_hud.IsVisible()); break;
    }
}

//------------------------------------------------------------------------------
static void
callbackKernel(int k) {

    g_kernel = k;
}

static void
callbackLevel(int l) {

    g_isolationLevel = l;

    rebuildMesh();
}


//------------------------------------------------------------------------------
static void
callbackAnimate(bool checked, int /* m */) {

    g_moveScale = checked;
}

//------------------------------------------------------------------------------
static void
callbackFreeze(bool checked, int /* f */) {

    g_freeze = checked;
}

//------------------------------------------------------------------------------
static void
callbackDisplayCageVertices(bool checked, int /* d */) {

    g_drawCageVertices = checked;
}

//------------------------------------------------------------------------------
static void
callbackDisplayCageEdges(bool checked, int /* d */) {
    g_drawCageEdges = checked;
}

static void
callbackBilinear(bool checked, int /* a */) {
    g_bilinear = checked;
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
initHUD() {

    int windowWidth = g_width, windowHeight = g_height,
        frameBufferWidth = g_width, frameBufferHeight = g_height;

    // window size might not match framebuffer size on a high DPI display
    glfwGetWindowSize(g_window, &windowWidth, &windowHeight);
    glfwGetFramebufferSize(g_window, &frameBufferWidth, &frameBufferHeight);

    g_hud.Init(windowWidth, windowHeight, frameBufferWidth, frameBufferHeight);

    g_hud.SetFrameBuffer(new GLFrameBuffer);

    g_hud.AddCheckBox("Cage Edges (H)", true, 10, 10, callbackDisplayCageEdges, 0, 'h');
    g_hud.AddCheckBox("Cage Verts (J)", true, 10, 30, callbackDisplayCageVertices, 0, 'j');
    g_hud.AddCheckBox("Animate vertices (M)", g_moveScale != 0, 10, 50, callbackAnimate, 0, 'm');
    g_hud.AddCheckBox("Freeze (spc)", false, 10, 70, callbackFreeze, 0, ' ');

    g_hud.AddCheckBox("Bilinear Stencils (`)", g_bilinear!=0, 10, 190, callbackBilinear, 0, '`');

    int compute_pulldown = g_hud.AddPullDown("Compute (K)", 250, 10, 300, callbackKernel, 'k');
    g_hud.AddPullDownButton(compute_pulldown, "CPU", kCPU);
#ifdef OPENSUBDIV_HAS_OPENMP
    g_hud.AddPullDownButton(compute_pulldown, "OpenMP", kOPENMP);
#endif
#ifdef OPENSUBDIV_HAS_TBB
    g_hud.AddPullDownButton(compute_pulldown, "TBB", kTBB);
#endif

    for (int i = 1; i < 11; ++i) {
        char level[16];
        sprintf(level, "Lv. %d", i);
        g_hud.AddRadioButton(3, level, i==g_isolationLevel, 10, 210+i*20, callbackLevel, i, '0'+(i%10));
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

    glGenVertexArrays(1, &g_cageVertexVAO);
    glGenVertexArrays(1, &g_cageEdgeVAO);
    glGenVertexArrays(1, &g_stencilsVAO);

    glGenBuffers(1, &g_cageVertexVBO);
    glGenBuffers(1, &g_cageEdgeVBO);
}

//------------------------------------------------------------------------------
static void
uninitGL() {

    glDeleteBuffers(1, &g_cageVertexVBO);
    glDeleteBuffers(1, &g_cageEdgeVBO);

    glDeleteVertexArrays(1, &g_cageVertexVAO);
    glDeleteVertexArrays(1, &g_cageEdgeVAO);
    glDeleteVertexArrays(1, &g_stencilsVAO);
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
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
#else
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3);
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
#endif
    glfwOpenWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
}

//------------------------------------------------------------------------------
int main(int argc, char **argv) {

    bool fullscreen = false;

    std::string str;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-d")) {
            g_isolationLevel = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-f")) {
            fullscreen = true;
        } else {
            std::ifstream ifs(argv[1]);
            if (ifs) {
                std::stringstream ss;
                ss << ifs.rdbuf();
                ifs.close();
                str = ss.str();
                g_defaultShapes.push_back(ShapeDesc(argv[1], str.c_str(), kCatmark));
            }
        }
    }

    initShapes();

    glfwSetErrorCallback(callbackErrorGLFW);
    if (not glfwInit()) {
        printf("Failed to initialize GLFW\n");
        return 1;
    }

    static const char windowTitle[] = "OpenSubdiv glStencilViewer " OPENSUBDIV_VERSION_STRING;

#define CORE_PROFILE
#ifdef CORE_PROFILE
    setGLCoreProfile();
#endif

    if (fullscreen) {

        g_primary = glfwGetPrimaryMonitor();

        // apparently glfwGetPrimaryMonitor fails under linux : if no primary,
        // settle for the first one in the list
        if (not g_primary) {
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

    if (not (g_window=glfwCreateWindow(g_width, g_height, windowTitle,
                                       fullscreen and g_primary ? g_primary : NULL, NULL))) {
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
