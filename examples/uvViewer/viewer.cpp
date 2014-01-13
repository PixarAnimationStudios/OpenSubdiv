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

#if defined(GLFW_VERSION_3)
    #include <GLFW/glfw3.h>
    GLFWwindow* g_window = 0;
    GLFWmonitor* g_primary = 0;
#else
    #include <GL/glfw.h>
#endif

#include <osd/error.h>
#include <osd/vertex.h>
#include <osd/glDrawContext.h>
#include <osd/glDrawRegistry.h>

#include <osd/cpuGLVertexBuffer.h>
#include <osd/cpuComputeContext.h>
#include <osd/cpuComputeController.h>
OpenSubdiv::OsdCpuComputeController *g_cpuComputeController = NULL;

#include <osd/glMesh.h>
OpenSubdiv::OsdGLMeshInterface *g_mesh = NULL;

#include <common/shape_utils.h>
#include "../common/stopwatch.h"
#include "../common/simple_math.h"
#include "../common/gl_hud.h"

static const char *shaderSource =
#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
    #include "shader.gen.h"
#else
    #include "shader_gl3.gen.h"
#endif
;

#include <cfloat>
#include <vector>
#include <fstream>
#include <sstream>
#include <utility>
#include <string>
#include <algorithm>

typedef OpenSubdiv::HbrMesh<OpenSubdiv::OsdVertex>     OsdHbrMesh;
typedef OpenSubdiv::HbrVertex<OpenSubdiv::OsdVertex>   OsdHbrVertex;
typedef OpenSubdiv::HbrFace<OpenSubdiv::OsdVertex>     OsdHbrFace;
typedef OpenSubdiv::HbrHalfedge<OpenSubdiv::OsdVertex> OsdHbrHalfedge;

enum DisplayStyle { kWire = 0,
                    kShaded,
                    kWireShaded };

struct SimpleShape {
    std::string  name;
    Scheme       scheme;
    std::string  data;

    SimpleShape() { }
    SimpleShape(std::string const & idata, char const * iname, Scheme ischeme)
        : name(iname), scheme(ischeme), data(idata) { }
};

std::vector<SimpleShape> g_defaultShapes;

int g_currentShape = 0;

int   g_frame = 0,
      g_repeatCount = 0;
int   g_fvarBoundary = OsdHbrMesh::k_InterpolateBoundaryEdgeOnly;
int   g_fvarPropagateCorners = 0;

// GUI variables
int   g_fullscreen = 0,
      g_freeze = 0,
      g_displayStyle = kWireShaded,
      g_adaptive = 0,
      g_mbutton[3] = {0, 0, 0},
      g_mouseUvView = 0,
      g_running = 1;

float g_moveScale = 0.0f;

float g_rotate[2] = {0, 0},
      g_dolly = 5,
      g_pan[2] = {0, 0},
      g_center[3] = {0, 0, 0},
      g_size = 0,
      g_uvPan[2] = {0, 0},
      g_uvScale = 1.0;

int   g_prev_x = 0,
      g_prev_y = 0;

int   g_width = 1600,
      g_height = 800;

GLhud g_hud;

// geometry
std::vector<float> g_orgPositions,
                   g_positions,
                   g_normals;

Scheme             g_scheme;

int g_level = 2;
int g_tessLevel = 1;
int g_tessLevelMin = 1;

GLuint g_transformUB = 0,
       g_transformBinding = 0,
       g_tessellationUB = 0,
       g_tessellationBinding = 0;

struct Transform {
    float ModelViewMatrix[16];
    float ProjectionMatrix[16];
    float ModelViewProjectionMatrix[16];
    float ModelViewInverseMatrix[16];
    float UvViewMatrix[16];
} g_transformData;

GLuint g_vao = 0;
GLuint g_cageEdgeVAO = 0,
       g_cageEdgeVBO = 0,
       g_cageVertexVAO = 0,
       g_cageVertexVBO = 0;

std::vector<int> g_coarseEdges;
std::vector<float> g_coarseEdgeSharpness;
std::vector<float> g_coarseVertexSharpness;

struct Program
{
    GLuint program;
    GLuint uniformModelViewProjectionMatrix;
    GLuint attrPosition;
    GLuint attrColor;
} g_defaultProgram;

//------------------------------------------------------------------------------
static GLuint
compileShader(GLenum shaderType, const char *source)
{
    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    return shader;
}

static bool
linkDefaultProgram()
{
#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
    #define GLSL_VERSION_DEFINE "#version 400\n"
#else
    #define GLSL_VERSION_DEFINE "#version 150\n"
#endif

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

    GLuint program = glCreateProgram();
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vsSrc);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fsSrc);

    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);

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
    g_defaultProgram.uniformModelViewProjectionMatrix =
        glGetUniformLocation(program, "ModelViewProjectionMatrix");
    g_defaultProgram.attrPosition = glGetAttribLocation(program, "position");
    g_defaultProgram.attrColor = glGetAttribLocation(program, "color");

    return true;
}

//------------------------------------------------------------------------------
static void
initializeShapes()
{
#include <shapes/catmark_cube_corner0.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube_corner0, "catmark_cube_corner0", kCatmark));

#include <shapes/catmark_cube_corner1.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube_corner1, "catmark_cube_corner1", kCatmark));

#include <shapes/catmark_cube_corner2.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube_corner2, "catmark_cube_corner2", kCatmark));

#include <shapes/catmark_cube_corner3.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube_corner3, "catmark_cube_corner3", kCatmark));

#include <shapes/catmark_cube_corner4.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube_corner4, "catmark_cube_corner4", kCatmark));

#include <shapes/catmark_cube_creases0.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube_creases0, "catmark_cube_creases0", kCatmark));

#include <shapes/catmark_cube_creases1.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube_creases1, "catmark_cube_creases1", kCatmark));

#include <shapes/catmark_cube.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube, "catmark_cube", kCatmark));

#include <shapes/catmark_dart_edgecorner.h>
    g_defaultShapes.push_back(SimpleShape(catmark_dart_edgecorner, "catmark_dart_edgecorner", kCatmark));

#include <shapes/catmark_dart_edgeonly.h>
    g_defaultShapes.push_back(SimpleShape(catmark_dart_edgeonly, "catmark_dart_edgeonly", kCatmark));

#include <shapes/catmark_edgecorner.h>
    g_defaultShapes.push_back(SimpleShape(catmark_edgecorner, "catmark_edgecorner", kCatmark));

#include <shapes/catmark_edgeonly.h>
    g_defaultShapes.push_back(SimpleShape(catmark_edgeonly, "catmark_edgeonly", kCatmark));

#include <shapes/catmark_gregory_test1.h>
    g_defaultShapes.push_back(SimpleShape(catmark_gregory_test1, "catmark_gregory_test1", kCatmark));

#include <shapes/catmark_gregory_test2.h>
    g_defaultShapes.push_back(SimpleShape(catmark_gregory_test2, "catmark_gregory_test2", kCatmark));

#include <shapes/catmark_gregory_test3.h>
    g_defaultShapes.push_back(SimpleShape(catmark_gregory_test3, "catmark_gregory_test3", kCatmark));

#include <shapes/catmark_gregory_test4.h>
    g_defaultShapes.push_back(SimpleShape(catmark_gregory_test4, "catmark_gregory_test4", kCatmark));

#include <shapes/catmark_hole_test1.h>
    g_defaultShapes.push_back(SimpleShape(catmark_hole_test1, "catmark_hole_test1", kCatmark));

#include <shapes/catmark_hole_test2.h>
    g_defaultShapes.push_back(SimpleShape(catmark_hole_test2, "catmark_hole_test2", kCatmark));

#include <shapes/catmark_pyramid_creases0.h>
    g_defaultShapes.push_back(SimpleShape(catmark_pyramid_creases0, "catmark_pyramid_creases0", kCatmark));

#include <shapes/catmark_pyramid_creases1.h>
    g_defaultShapes.push_back(SimpleShape(catmark_pyramid_creases1, "catmark_pyramid_creases1", kCatmark));

#include <shapes/catmark_pyramid.h>
    g_defaultShapes.push_back(SimpleShape(catmark_pyramid, "catmark_pyramid", kCatmark));

#include <shapes/catmark_tent_creases0.h>
    g_defaultShapes.push_back(SimpleShape(catmark_tent_creases0, "catmark_tent_creases0", kCatmark));

#include <shapes/catmark_tent_creases1.h>
    g_defaultShapes.push_back(SimpleShape(catmark_tent_creases1, "catmark_tent_creases1", kCatmark));

#include <shapes/catmark_tent.h>
    g_defaultShapes.push_back(SimpleShape(catmark_tent, "catmark_tent", kCatmark));

#include <shapes/catmark_torus.h>
    g_defaultShapes.push_back(SimpleShape(catmark_torus, "catmark_torus", kCatmark));

#include <shapes/catmark_torus_creases0.h>
    g_defaultShapes.push_back(SimpleShape(catmark_torus_creases0, "catmark_torus_creases0", kCatmark));

#include <shapes/catmark_square_hedit0.h>
    g_defaultShapes.push_back(SimpleShape(catmark_square_hedit0, "catmark_square_hedit0", kCatmark));

#include <shapes/catmark_square_hedit1.h>
    g_defaultShapes.push_back(SimpleShape(catmark_square_hedit1, "catmark_square_hedit1", kCatmark));

#include <shapes/catmark_square_hedit2.h>
    g_defaultShapes.push_back(SimpleShape(catmark_square_hedit2, "catmark_square_hedit2", kCatmark));

#include <shapes/catmark_square_hedit3.h>
    g_defaultShapes.push_back(SimpleShape(catmark_square_hedit3, "catmark_square_hedit3", kCatmark));

#include <shapes/catmark_square_hedit4.h>
    g_defaultShapes.push_back(SimpleShape(catmark_square_hedit4, "catmark_square_hedit4", kCatmark));

#include <shapes/catmark_bishop.h>
    g_defaultShapes.push_back(SimpleShape(catmark_bishop, "catmark_bishop", kCatmark));

#include <shapes/catmark_car.h>
    g_defaultShapes.push_back(SimpleShape(catmark_car, "catmark_car", kCatmark));

#include <shapes/catmark_helmet.h>
    g_defaultShapes.push_back(SimpleShape(catmark_helmet, "catmark_helmet", kCatmark));

#include <shapes/catmark_pawn.h>
    g_defaultShapes.push_back(SimpleShape(catmark_pawn, "catmark_pawn", kCatmark));

#include <shapes/catmark_rook.h>
    g_defaultShapes.push_back(SimpleShape(catmark_rook, "catmark_rook", kCatmark));

#include <shapes/bilinear_cube.h>
    g_defaultShapes.push_back(SimpleShape(bilinear_cube, "bilinear_cube", kBilinear));


#include <shapes/loop_cube_creases0.h>
    g_defaultShapes.push_back(SimpleShape(loop_cube_creases0, "loop_cube_creases0", kLoop));

#include <shapes/loop_cube_creases1.h>
    g_defaultShapes.push_back(SimpleShape(loop_cube_creases1, "loop_cube_creases1", kLoop));

#include <shapes/loop_cube.h>
    g_defaultShapes.push_back(SimpleShape(loop_cube, "loop_cube", kLoop));

#include <shapes/loop_icosahedron.h>
    g_defaultShapes.push_back(SimpleShape(loop_icosahedron, "loop_icosahedron", kLoop));

#include <shapes/loop_saddle_edgecorner.h>
    g_defaultShapes.push_back(SimpleShape(loop_saddle_edgecorner, "loop_saddle_edgecorner", kLoop));

#include <shapes/loop_saddle_edgeonly.h>
    g_defaultShapes.push_back(SimpleShape(loop_saddle_edgeonly, "loop_saddle_edgeonly", kLoop));

#include <shapes/loop_triangle_edgecorner.h>
    g_defaultShapes.push_back(SimpleShape(loop_triangle_edgecorner, "loop_triangle_edgecorner", kLoop));

#include <shapes/loop_triangle_edgeonly.h>
    g_defaultShapes.push_back(SimpleShape(loop_triangle_edgeonly, "loop_triangle_edgeonly", kLoop));
}

//------------------------------------------------------------------------------
static void
calcNormals(OsdHbrMesh * mesh, std::vector<float> const & pos, std::vector<float> & result)
{
    // calc normal vectors
    int nverts = (int)pos.size()/3;

    int nfaces = mesh->GetNumCoarseFaces();
    for (int i = 0; i < nfaces; ++i) {
        OsdHbrFace * f = mesh->GetFace(i);

        float const * p0 = &pos[f->GetVertex(0)->GetID()*3],
                    * p1 = &pos[f->GetVertex(1)->GetID()*3],
                    * p2 = &pos[f->GetVertex(2)->GetID()*3];

        float n[3];
        cross(n, p0, p1, p2);

        for (int j = 0; j < f->GetNumVertices(); j++) {
            int idx = f->GetVertex(j)->GetID() * 3;
            result[idx  ] += n[0];
            result[idx+1] += n[1];
            result[idx+2] += n[2];
        }
    }
    for (int i = 0; i < nverts; ++i)
        normalize(&result[i*3]);
}

//------------------------------------------------------------------------------
static void
updateGeom()
{
    int nverts = (int)g_orgPositions.size() / 3;

    std::vector<float> vertex;
    vertex.reserve(nverts*3);

    const float *p = &g_orgPositions[0];

    float r = sin(g_frame*0.001f) * g_moveScale;
    for (int i = 0; i < nverts; ++i) {
        float ct = cos(p[2] * r);
        float st = sin(p[2] * r);
        g_positions[i*3+0] = p[0]*ct + p[1]*st;
        g_positions[i*3+1] = -p[0]*st + p[1]*ct;
        g_positions[i*3+2] = p[2];

        p += 3;
    }

    p = &g_orgPositions[0];
    const float *pp = &g_positions[0];
    for (int i = 0; i < nverts; ++i) {
        vertex.push_back(pp[0]);
        vertex.push_back(pp[1]);
        vertex.push_back(pp[2]);
        pp += 3;
    }

    g_mesh->UpdateVertexBuffer(&vertex[0], 0, nverts);

    g_mesh->Refine();

    g_mesh->Synchronize();
}

//------------------------------------------------------------------------------
static void
createOsdMesh(const std::string &shape, int level, Scheme scheme = kCatmark)
{
    // generate Hbr representation from "obj" description
    OsdHbrMesh * hmesh = simpleHbr<OpenSubdiv::OsdVertex>(shape.c_str(),
                                                          scheme,
                                                          g_orgPositions,
                                                          /*fvar*/ true);

    g_normals.resize(g_orgPositions.size(), 0.0f);
    g_positions.resize(g_orgPositions.size(), 0.0f);
    calcNormals(hmesh, g_orgPositions, g_normals);

    // save coarse topology (used for coarse mesh drawing)
    g_coarseEdges.clear();
    g_coarseEdgeSharpness.clear();
    g_coarseVertexSharpness.clear();
    int nf = hmesh->GetNumFaces();
    for (int i = 0; i < nf; ++i) {
        OsdHbrFace *face = hmesh->GetFace(i);
        int nv = face->GetNumVertices();
        for (int j = 0; j < nv; ++j) {
            g_coarseEdges.push_back(face->GetVertex(j)->GetID());
            g_coarseEdges.push_back(face->GetVertex((j+1)%nv)->GetID());
            g_coarseEdgeSharpness.push_back(face->GetEdge(j)->GetSharpness());
        }
    }
    int nv = hmesh->GetNumVertices();
    for (int i = 0; i < nv; ++i) {
        g_coarseVertexSharpness.push_back(hmesh->GetVertex(i)->GetSharpness());
    }

    hmesh->SetFVarInterpolateBoundaryMethod((OsdHbrMesh::InterpolateBoundaryMethod)g_fvarBoundary);
    hmesh->SetFVarPropagateCorners(g_fvarPropagateCorners != 0);

    delete g_mesh;
    g_mesh = NULL;

    g_scheme = scheme;

    // Adaptive refinement currently supported only for catmull-clark scheme
    bool doAdaptive = (g_adaptive != 0 and g_scheme == kCatmark);

    OpenSubdiv::OsdMeshBitset bits;
    bits.set(OpenSubdiv::MeshAdaptive, doAdaptive);
    bits.set(OpenSubdiv::MeshFVarData, 1);

    int numVertexElements = 3;
    int numVaryingElements = 0;

    if (not g_cpuComputeController) {
        g_cpuComputeController = new OpenSubdiv::OsdCpuComputeController();
    }
    g_mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdCpuGLVertexBuffer,
        OpenSubdiv::OsdCpuComputeController,
        OpenSubdiv::OsdGLDrawContext>(
            g_cpuComputeController,
            hmesh,
            numVertexElements,
            numVaryingElements,
            level, bits);

    // Hbr mesh can be deleted
    delete hmesh;

    // compute model bounding
    float min[3] = { FLT_MAX,  FLT_MAX,  FLT_MAX};
    float max[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (size_t i = 0; i < g_orgPositions.size()/3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float v = g_orgPositions[i*3+j];
            min[j] = std::min(min[j], v);
            max[j] = std::max(max[j], v);
        }
    }
    for (int j = 0; j < 3; ++j) {
        g_center[j] = (min[j] + max[j]) * 0.5f;
        g_size += (max[j]-min[j])*(max[j]-min[j]);
    }
    g_size = sqrtf(g_size);

    updateGeom();

    // -------- VAO
    glBindVertexArray(g_vao);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_mesh->GetDrawContext()->GetPatchIndexBuffer());
    glBindBuffer(GL_ARRAY_BUFFER, g_mesh->BindVertexBuffer());

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 3, 0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

//------------------------------------------------------------------------------
static void
fitFrame()
{
    g_pan[0] = g_pan[1] = 0;
    g_dolly = g_size;
    g_uvPan[0] = g_uvPan[1] = 0;
    g_uvScale = 1.0;
}

//------------------------------------------------------------------------------
static inline void
setSharpnessColor(float s, float *r, float *g, float *b)
{
    //  0.0       2.0       4.0
    // green --- yellow --- red
    *r = std::min(1.0f, s * 0.5f);
    *g = std::min(1.0f, 2.0f - s*0.5f);
    *b = 0;
}

static void
drawCageEdges()
{
    glUseProgram(g_defaultProgram.program);
    glUniformMatrix4fv(g_defaultProgram.uniformModelViewProjectionMatrix,
                       1, GL_FALSE, g_transformData.ModelViewProjectionMatrix);

    std::vector<float> vbo;
    vbo.reserve(g_coarseEdges.size() * 6);
    float r, g, b;
    for (int i = 0; i < (int)g_coarseEdges.size(); i+=2) {
        setSharpnessColor(g_coarseEdgeSharpness[i/2], &r, &g, &b);
        for (int j = 0; j < 2; ++j) {
            vbo.push_back(g_positions[g_coarseEdges[i+j]*3]);
            vbo.push_back(g_positions[g_coarseEdges[i+j]*3+1]);
            vbo.push_back(g_positions[g_coarseEdges[i+j]*3+2]);
            vbo.push_back(r);
            vbo.push_back(g);
            vbo.push_back(b);
        }
    }

    glBindVertexArray(g_cageEdgeVAO);

    glBindBuffer(GL_ARRAY_BUFFER, g_cageEdgeVBO);
    glBufferData(GL_ARRAY_BUFFER, (int)vbo.size() * sizeof(float), &vbo[0],
                 GL_STATIC_DRAW);

    glEnableVertexAttribArray(g_defaultProgram.attrPosition);
    glEnableVertexAttribArray(g_defaultProgram.attrColor);
    glVertexAttribPointer(g_defaultProgram.attrPosition,
                          3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 6, 0);
    glVertexAttribPointer(g_defaultProgram.attrColor,
                          3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 6, (void*)12);

    glDrawArrays(GL_LINES, 0, (int)g_coarseEdges.size());

    glBindVertexArray(0);
    glUseProgram(0);
}

static void
drawCageVertices()
{
    glUseProgram(g_defaultProgram.program);
    glUniformMatrix4fv(g_defaultProgram.uniformModelViewProjectionMatrix,
                       1, GL_FALSE, g_transformData.ModelViewProjectionMatrix);

    int numPoints = (int)g_positions.size()/3;
    std::vector<float> vbo;
    vbo.reserve(numPoints*6);
    float r, g, b;
    for (int i = 0; i < numPoints; ++i) {
        setSharpnessColor(g_coarseVertexSharpness[i], &r, &g, &b);
        vbo.push_back(g_positions[i*3+0]);
        vbo.push_back(g_positions[i*3+1]);
        vbo.push_back(g_positions[i*3+2]);
        vbo.push_back(r);
        vbo.push_back(g);
        vbo.push_back(b);
    }

    glBindVertexArray(g_cageVertexVAO);

    glBindBuffer(GL_ARRAY_BUFFER, g_cageVertexVBO);
    glBufferData(GL_ARRAY_BUFFER, (int)vbo.size() * sizeof(float), &vbo[0],
                 GL_STATIC_DRAW);

    glEnableVertexAttribArray(g_defaultProgram.attrPosition);
    glEnableVertexAttribArray(g_defaultProgram.attrColor);
    glVertexAttribPointer(g_defaultProgram.attrPosition,
                          3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 6, 0);
    glVertexAttribPointer(g_defaultProgram.attrColor,
                          3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 6, (void*)12);

    glPointSize(10.0f);
    glDrawArrays(GL_POINTS, 0, numPoints);
    glPointSize(1.0f);

    glBindVertexArray(0);
    glUseProgram(0);
}

//------------------------------------------------------------------------------

union Effect
{
    Effect(int displayStyle_, int uvDraw_) : value(0) {
        displayStyle = displayStyle_;
        uvDraw = uvDraw_;
    }

    struct {
        unsigned int displayStyle:3;
        unsigned int uvDraw:1;
    };
    int value;

    bool operator < (const Effect &e) const {
        return value < e.value;
    }
};

typedef std::pair<OpenSubdiv::OsdDrawContext::PatchDescriptor, Effect> EffectDesc;

class EffectDrawRegistry : public OpenSubdiv::OsdGLDrawRegistry<EffectDesc>
{
  protected:
    virtual ConfigType *
    _CreateDrawConfig(DescType const & desc, SourceConfigType const * sconfig);

    virtual SourceConfigType *
    _CreateDrawSourceConfig(DescType const & desc);
};

EffectDrawRegistry::SourceConfigType *
EffectDrawRegistry::_CreateDrawSourceConfig(DescType const & desc)
{
    Effect effect = desc.second;

    SourceConfigType * sconfig =
        BaseRegistry::_CreateDrawSourceConfig(desc.first);

    assert(sconfig);

#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
    const char *glslVersion = "#version 400\n";
#else
    const char *glslVersion = "#version 330\n";
#endif

    if (desc.first.GetType() == OpenSubdiv::FarPatchTables::QUADS or
        desc.first.GetType() == OpenSubdiv::FarPatchTables::TRIANGLES) {
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

    sconfig->commonShader.AddDefine("OSD_FVAR_WIDTH", "2");


    if (desc.first.GetType() == OpenSubdiv::FarPatchTables::QUADS) {
        // uniform catmark, bilinear
        sconfig->geometryShader.AddDefine("PRIM_QUAD");
        sconfig->fragmentShader.AddDefine("PRIM_QUAD");
        sconfig->commonShader.AddDefine("UNIFORM_SUBDIVISION");
    } else if (desc.first.GetType() == OpenSubdiv::FarPatchTables::TRIANGLES) {
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

    if (effect.uvDraw) {
        sconfig->commonShader.AddDefine("GEOMETRY_OUT_FILL");
        sconfig->commonShader.AddDefine("GEOMETRY_UV_VIEW");
    } else {
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
        }
    }

    return sconfig;
}

EffectDrawRegistry::ConfigType *
EffectDrawRegistry::_CreateDrawConfig(
        DescType const & desc,
        SourceConfigType const * sconfig)
{
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

    GLint loc;
#if not defined(GL_ARB_separate_shader_objects) || defined(GL_VERSION_4_1)
    glUseProgram(config->program);
    if ((loc = glGetUniformLocation(config->program, "OsdVertexBuffer")) != -1) {
        glUniform1i(loc, 0);  // GL_TEXTURE0
    }
    if ((loc = glGetUniformLocation(config->program, "OsdValenceBuffer")) != -1) {
        glUniform1i(loc, 1);  // GL_TEXTURE1
    }
    if ((loc = glGetUniformLocation(config->program, "OsdQuadOffsetBuffer")) != -1) {
        glUniform1i(loc, 2);  // GL_TEXTURE2
    }
    if ((loc = glGetUniformLocation(config->program, "OsdPatchParamBuffer")) != -1) {
        glUniform1i(loc, 3);  // GL_TEXTURE3
    }
    if ((loc = glGetUniformLocation(config->program, "OsdFVarDataBuffer")) != -1) {
        glUniform1i(loc, 4);  // GL_TEXTURE4
    }
#else
    if ((loc = glGetUniformLocation(config->program, "OsdVertexBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 0);  // GL_TEXTURE0
    }
    if ((loc = glGetUniformLocation(config->program, "OsdValenceBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 1);  // GL_TEXTURE1
    }
    if ((loc = glGetUniformLocation(config->program, "OsdQuadOffsetBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 2);  // GL_TEXTURE2
    }
    if ((loc = glGetUniformLocation(config->program, "OsdPatchParamBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 3);  // GL_TEXTURE3
    }
    if ((loc = glGetUniformLocation(config->program, "OsdFVarDataBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 4);  // GL_TEXTURE4
    }
#endif

    return config;
}

EffectDrawRegistry effectRegistry;

static Effect
GetEffect(bool uvDraw = false)
{
    return Effect(g_displayStyle, uvDraw);
}

//------------------------------------------------------------------------------
static GLuint
bindProgram(Effect effect, OpenSubdiv::OsdDrawContext::PatchArray const & patch)
{
    EffectDesc effectDesc(patch.GetDescriptor(), effect);
    EffectDrawRegistry::ConfigType *
        config = effectRegistry.GetDrawConfig(effectDesc);

    GLuint program = config->program;

    glUseProgram(program);

    if (!g_transformUB) {
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

    if (!g_tessellationUB) {
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

    if (g_mesh->GetDrawContext()->GetVertexTextureBuffer()) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_BUFFER,
            g_mesh->GetDrawContext()->GetVertexTextureBuffer());
    }
    if (g_mesh->GetDrawContext()->GetVertexValenceTextureBuffer()) {
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_BUFFER,
            g_mesh->GetDrawContext()->GetVertexValenceTextureBuffer());
    }
    if (g_mesh->GetDrawContext()->GetQuadOffsetsTextureBuffer()) {
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_BUFFER,
            g_mesh->GetDrawContext()->GetQuadOffsetsTextureBuffer());
    }
    if (g_mesh->GetDrawContext()->GetPatchParamTextureBuffer()) {
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_BUFFER,
            g_mesh->GetDrawContext()->GetPatchParamTextureBuffer());
    }
    if (g_mesh->GetDrawContext()->GetFvarDataTextureBuffer()) {
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_BUFFER,
            g_mesh->GetDrawContext()->GetFvarDataTextureBuffer());
    }

    glActiveTexture(GL_TEXTURE0);

    return program;
}

//------------------------------------------------------------------------------
static void
display()
{
    Stopwatch s;
    s.Start();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // ---------------------------------------------
    glViewport(0, 0, g_width/2, g_height);

    // prepare view matrix
    double aspect = (g_width/2)/(double)g_height;
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

    identity(g_transformData.UvViewMatrix);
    scale(g_transformData.UvViewMatrix, g_uvScale, g_uvScale, 1);
    translate(g_transformData.UvViewMatrix, -g_uvPan[0], -g_uvPan[1], 0);

    // make sure that the vertex buffer is interoped back as a GL resources.
    g_mesh->BindVertexBuffer();

    glBindVertexArray(g_vao);

    OpenSubdiv::OsdDrawContext::PatchArrayVector const & patches =
        g_mesh->GetDrawContext()->patchArrays;

    if (g_displayStyle == kWire)
        glDisable(GL_CULL_FACE);

    // patch drawing
    for (int i = 0; i < (int)patches.size(); ++i) {
        OpenSubdiv::OsdDrawContext::PatchArray const & patch = patches[i];

        OpenSubdiv::OsdDrawContext::PatchDescriptor desc = patch.GetDescriptor();
        OpenSubdiv::FarPatchTables::Type patchType = desc.GetType();

        GLenum primType;

        switch (patchType) {
        case OpenSubdiv::FarPatchTables::QUADS:
            primType = GL_LINES_ADJACENCY;
            break;
        case OpenSubdiv::FarPatchTables::TRIANGLES:
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
        glDrawElements(primType, patch.GetNumIndices(), GL_UNSIGNED_INT,
                       (void *)(patch.GetVertIndex() * sizeof(unsigned int)));
    }
    if (g_displayStyle == kWire)
        glEnable(GL_CULL_FACE);

    glBindVertexArray(0);
    glUseProgram(0);

    drawCageEdges();
    drawCageVertices();

    // ---------------------------------------------
    // uv viewport
    glViewport(g_width/2, 0, g_width/2, g_height);

    g_mesh->BindVertexBuffer();

    glBindVertexArray(g_vao);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    for (int i = 0; i < (int)patches.size(); ++i) {
        OpenSubdiv::OsdDrawContext::PatchArray const & patch = patches[i];

        OpenSubdiv::OsdDrawContext::PatchDescriptor desc = patch.GetDescriptor();
        OpenSubdiv::FarPatchTables::Type patchType = desc.GetType();

        GLenum primType;

        switch (patchType) {
        case OpenSubdiv::FarPatchTables::QUADS:
            primType = GL_LINES_ADJACENCY;
            break;
        case OpenSubdiv::FarPatchTables::TRIANGLES:
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
        GLuint program = bindProgram(GetEffect(/*uvDraw=*/ true), patch);

        GLuint uniformGregoryQuadOffsetBase =
          glGetUniformLocation(program, "GregoryQuadOffsetBase");
        GLuint uniformPrimitiveIdBase =
          glGetUniformLocation(program, "PrimitiveIdBase");

        glProgramUniform1i(program, uniformGregoryQuadOffsetBase,
                           patch.GetQuadOffsetIndex());
        glProgramUniform1i(program, uniformPrimitiveIdBase,
                           patch.GetPatchIndex());
#else
        GLuint program = bindProgram(GetEffect(/*uvDraw=*/ true), patch);
        GLint uniformPrimitiveIdBase =
          glGetUniformLocation(program, "PrimitiveIdBase");
        if (uniformPrimitiveIdBase != -1)
            glUniform1i(uniformPrimitiveIdBase, patch.GetPatchIndex());
#endif
        glDrawElements(primType, patch.GetNumIndices(), GL_UNSIGNED_INT,
                       (void *)(patch.GetVertIndex() * sizeof(unsigned int)));
    }

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    // full viewport
    glViewport(0, 0, g_width, g_height);

    if (g_hud.IsVisible()) {
        g_hud.DrawString(10, -40, "Tess level : %d", g_tessLevel);
        g_hud.Flush();
    }

    glFinish();
}

//------------------------------------------------------------------------------
static void
#if GLFW_VERSION_MAJOR >= 3
motion(GLFWwindow *, double dx, double dy)
{
    int x=(int)dx, y=(int)dy;
#else
motion(int x, int y)
{
#endif
    if (g_mouseUvView) {
        if (!g_mbutton[0] && !g_mbutton[1] && g_mbutton[2]) {
            // pan
            g_uvPan[0] -= (x - g_prev_x) * 2 / g_uvScale / static_cast<float>(g_width/2);
            g_uvPan[1] += (y - g_prev_y) * 2 / g_uvScale / static_cast<float>(g_height);
        } else if ((g_mbutton[0] && !g_mbutton[1] && g_mbutton[2]) or
                   (!g_mbutton[0] && g_mbutton[1] && !g_mbutton[2])) {
            // scale
            g_uvScale += g_uvScale*0.01f*(x - g_prev_x);
            g_uvScale = std::max(std::min(g_uvScale, 100.0f), 0.01f);
        }
    } else {
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
            if (g_dolly <= 0.01) g_dolly = 0.01f;
        }
    }

    g_prev_x = x;
    g_prev_y = y;
}

//------------------------------------------------------------------------------
static void
#if GLFW_VERSION_MAJOR >= 3
mouse(GLFWwindow *, int button, int state, int mods)
#else
mouse(int button, int state)
#endif
{
    if (button == 0 && state == GLFW_PRESS && g_hud.MouseClick(g_prev_x, g_prev_y))
        return;

    if (button < 3) {
        g_mbutton[button] = (state == GLFW_PRESS);
    }

    g_mouseUvView = (g_prev_x > g_width/2);
}

//------------------------------------------------------------------------------
static void
uninitGL()
{
    glDeleteBuffers(1, &g_cageVertexVBO);
    glDeleteBuffers(1, &g_cageEdgeVBO);
    glDeleteVertexArrays(1, &g_vao);
    glDeleteVertexArrays(1, &g_cageVertexVAO);
    glDeleteVertexArrays(1, &g_cageEdgeVAO);

    if (g_mesh)
        delete g_mesh;

    delete g_cpuComputeController;
}

//------------------------------------------------------------------------------
static void
#if GLFW_VERSION_MAJOR >= 3
reshape(GLFWwindow *, int width, int height)
#else
reshape(int width, int height)
#endif
{
    g_width = width;
    g_height = height;

    g_hud.Rebuild(width, height);
}

//------------------------------------------------------------------------------
#if GLFW_VERSION_MAJOR >= 3
void windowClose(GLFWwindow*)
{
    g_running = false;
}
#else
int windowClose()
{
    g_running = false;
    return GL_TRUE;
}
#endif

//------------------------------------------------------------------------------
static void
toggleFullScreen() {
    // XXXX manuelk : to re-implement from glut
}

//------------------------------------------------------------------------------
static void
#if GLFW_VERSION_MAJOR >= 3
keyboard(GLFWwindow *, int key, int scancode, int event, int mods)
#else
#define GLFW_KEY_ESCAPE GLFW_KEY_ESC
keyboard(int key, int event)
#endif
{
    if (event == GLFW_RELEASE) return;
    if (g_hud.KeyDown(tolower(key))) return;

    switch (key) {
        case 'Q': g_running = 0; break;
        case 'F': fitFrame(); break;
        case GLFW_KEY_TAB: toggleFullScreen(); break;
        case '+':
        case '=':  g_tessLevel++; break;
        case '-':  g_tessLevel = std::max(g_tessLevelMin, g_tessLevel-1); break;
        case GLFW_KEY_ESCAPE: g_hud.SetVisible(!g_hud.IsVisible()); break;
    }
}

//------------------------------------------------------------------------------
static void
rebuildOsdMesh()
{
    createOsdMesh(g_defaultShapes[g_currentShape].data,
                  g_level,
                  g_defaultShapes[g_currentShape].scheme);
}

static void
callbackDisplayStyle(int b)
{
    g_displayStyle = b;
}

static void
callbackLevel(int l)
{
    g_level = l;
    rebuildOsdMesh();
}

static void
callbackModel(int m)
{
    int maxShapes = static_cast<int>(g_defaultShapes.size());
    g_currentShape = std::max(0, std::min(m, maxShapes-1));
    rebuildOsdMesh();
}

static void
callbackAdaptive(bool checked, int a)
{
    if (OpenSubdiv::OsdGLDrawContext::SupportsAdaptiveTessellation()) {
        g_adaptive = checked;
        rebuildOsdMesh();
    }
}

static void
callbackBoundary(int b)
{
    g_fvarBoundary = b;
    rebuildOsdMesh();
}

static void
callbackPropagateCorners(bool b, int button)
{
    g_fvarPropagateCorners = b;
    rebuildOsdMesh();
}

static void
initHUD()
{
    g_hud.Init(g_width, g_height);

    g_hud.AddRadioButton(1, "Wire (W)",    g_displayStyle == kWire,
                         200, 10, callbackDisplayStyle, 0, 'w');
    g_hud.AddRadioButton(1, "Shaded",      g_displayStyle == kShaded,
                         200, 30, callbackDisplayStyle, 1, 'w');
    g_hud.AddRadioButton(1, "Wire+Shaded", g_displayStyle == kWireShaded,
                         200, 50, callbackDisplayStyle, 2, 'w');

    if (OpenSubdiv::OsdGLDrawContext::SupportsAdaptiveTessellation())
        g_hud.AddCheckBox("Adaptive (`)", g_adaptive != 0, 10, 10, callbackAdaptive, 0, '`');

    for (int i = 1; i < 11; ++i) {
        char level[16];
        sprintf(level, "Lv. %d", i);
        g_hud.AddRadioButton(3, level, i == 2, 10, 20 + i*20, callbackLevel, i, '0'+(i%10));
    }

    for (int i = 0; i < (int)g_defaultShapes.size(); ++i) {
        g_hud.AddRadioButton(4, g_defaultShapes[i].name.c_str(),
                             i == g_currentShape, -220, 10+i*16, callbackModel, i, 'n');
    }

    g_hud.AddRadioButton(2, "Boundary none (B)",
                         g_fvarBoundary == OsdHbrMesh::k_InterpolateBoundaryNone,
                         350, 10, callbackBoundary, OsdHbrMesh::k_InterpolateBoundaryNone, 'b');
    g_hud.AddRadioButton(2, "Boundary edge only",
                         g_fvarBoundary == OsdHbrMesh::k_InterpolateBoundaryEdgeOnly,
                         350, 30, callbackBoundary, OsdHbrMesh::k_InterpolateBoundaryEdgeOnly, 'b');
    g_hud.AddRadioButton(2, "Boundary edge and corners",
                         g_fvarBoundary == OsdHbrMesh::k_InterpolateBoundaryEdgeAndCorner,
                         350, 50, callbackBoundary, OsdHbrMesh::k_InterpolateBoundaryEdgeAndCorner, 'b');
    g_hud.AddRadioButton(2, "Boundary always sharp",
                         g_fvarBoundary == OsdHbrMesh::k_InterpolateBoundaryAlwaysSharp,
                         350, 70, callbackBoundary, OsdHbrMesh::k_InterpolateBoundaryAlwaysSharp, 'b');

    g_hud.AddCheckBox("Propagate corners (C)", g_fvarPropagateCorners != 0,
                      350, 110, callbackPropagateCorners, 0, 'c');
}

//------------------------------------------------------------------------------
static void
initGL()
{
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);

    glGenVertexArrays(1, &g_vao);
    glGenVertexArrays(1, &g_cageVertexVAO);
    glGenVertexArrays(1, &g_cageEdgeVAO);
    glGenBuffers(1, &g_cageVertexVBO);
    glGenBuffers(1, &g_cageEdgeVBO);
}

//------------------------------------------------------------------------------
static void
idle()
{
    if (not g_freeze)
        g_frame++;

    updateGeom();

    if (g_repeatCount != 0 and g_frame >= g_repeatCount)
        g_running = 0;
}

//------------------------------------------------------------------------------
static void
callbackError(OpenSubdiv::OsdErrorType err, const char *message)
{
    printf("OsdError: %d\n", err);
    printf("%s", message);
}

//------------------------------------------------------------------------------
static void
setGLCoreProfile()
{
#if GLFW_VERSION_MAJOR >= 3
    #define glfwOpenWindowHint glfwWindowHint
    #define GLFW_OPENGL_VERSION_MAJOR GLFW_CONTEXT_VERSION_MAJOR
    #define GLFW_OPENGL_VERSION_MINOR GLFW_CONTEXT_VERSION_MINOR
#endif

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
                g_defaultShapes.push_back(SimpleShape(str.c_str(), argv[1], kCatmark));
            }
        }
    }
    initializeShapes();
    OsdSetErrorCallback(callbackError);

    if (not glfwInit()) {
        printf("Failed to initialize GLFW\n");
        return 1;
    }

    static const char windowTitle[] = "OpenSubdiv UV Viewer";

#define CORE_PROFILE
#ifdef CORE_PROFILE
    setGLCoreProfile();
#endif

#if GLFW_VERSION_MAJOR >= 3
    if (fullscreen) {
        g_primary = glfwGetPrimaryMonitor();

        // apparently glfwGetPrimaryMonitor fails under linux : if no primary,
        // settle for the first one in the list
        if (not g_primary) {
            int count = 0;
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
    glfwSetKeyCallback(g_window, keyboard);
    glfwSetCursorPosCallback(g_window, motion);
    glfwSetMouseButtonCallback(g_window, mouse);
    glfwSetWindowSizeCallback(g_window, reshape);
    glfwSetWindowCloseCallback(g_window, windowClose);
#else
    if (glfwOpenWindow(g_width, g_height, 8, 8, 8, 8, 24, 8,
                       fullscreen ? GLFW_FULLSCREEN : GLFW_WINDOW) == GL_FALSE) {
        printf("Failed to open window.\n");
        glfwTerminate();
        return 1;
    }
    glfwSetWindowTitle(windowTitle);
    glfwSetKeyCallback(keyboard);
    glfwSetMousePosCallback(motion);
    glfwSetMouseButtonCallback(mouse);
    glfwSetWindowSizeCallback(reshape);
    glfwSetWindowCloseCallback(windowClose);
#endif


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
    linkDefaultProgram();

    glfwSwapInterval(0);

    initHUD();
    rebuildOsdMesh();

    while (g_running) {
        idle();
        display();

#if GLFW_VERSION_MAJOR >= 3
        glfwPollEvents();
        glfwSwapBuffers(g_window);
#else
        glfwSwapBuffers();
#endif

        glFinish();
    }

    uninitGL();
    glfwTerminate();
}
