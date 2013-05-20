//
//     Copyright (C) Pixar. All rights reserved.
//
//     This license governs use of the accompanying software. If you
//     use the software, you accept this license. If you do not accept
//     the license, do not use the software.
//
//     1. Definitions
//     The terms "reproduce," "reproduction," "derivative works," and
//     "distribution" have the same meaning here as under U.S.
//     copyright law.  A "contribution" is the original software, or
//     any additions or changes to the software.
//     A "contributor" is any person or entity that distributes its
//     contribution under this license.
//     "Licensed patents" are a contributor's patent claims that read
//     directly on its contribution.
//
//     2. Grant of Rights
//     (A) Copyright Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free copyright license to reproduce its contribution,
//     prepare derivative works of its contribution, and distribute
//     its contribution or any derivative works that you create.
//     (B) Patent Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free license under its licensed patents to make, have
//     made, use, sell, offer for sale, import, and/or otherwise
//     dispose of its contribution in the software or derivative works
//     of the contribution in the software.
//
//     3. Conditions and Limitations
//     (A) No Trademark License- This license does not grant you
//     rights to use any contributor's name, logo, or trademarks.
//     (B) If you bring a patent claim against any contributor over
//     patents that you claim are infringed by the software, your
//     patent license from such contributor to the software ends
//     automatically.
//     (C) If you distribute any portion of the software, you must
//     retain all copyright, patent, trademark, and attribution
//     notices that are present in the software.
//     (D) If you distribute any portion of the software in source
//     code form, you may do so only under this license by including a
//     complete copy of this license with your distribution. If you
//     distribute any portion of the software in compiled or object
//     code form, you may only do so under a license that complies
//     with this license.
//     (E) The software is licensed "as-is." You bear the risk of
//     using it. The contributors give no express warranties,
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
//

#if defined(__APPLE__)
    #include <OpenGL/gl3.h>
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
    #include <GL/glfw3.h>
    GLFWwindow* g_window=0;
    GLFWmonitor* g_primary=0;
#else
    #include <GL/glfw.h>
#endif

#include <osd/cpuComputeContext.h>
#include <osd/cpuComputeController.h>
#include <osd/cpuEvalLimitContext.h>
#include <osd/cpuEvalLimitController.h>
#include <osd/cpuVertexBuffer.h>
#include <osd/cpuGLVertexBuffer.h>
#include <osd/error.h>
#include <osd/drawContext.h>
#include <osd/mesh.h>
#include <osd/vertex.h>

#include <common/shape_utils.h>
#include "../common/stopwatch.h"
#include "../common/simple_math.h"
#include "../common/gl_hud.h"

#include <cfloat>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdlib.h>

#ifdef OPENSUBDIV_HAS_OPENMP
    #include <omp.h>
#endif

using namespace OpenSubdiv;

//------------------------------------------------------------------------------
typedef HbrMesh<OsdVertex>     OsdHbrMesh;
typedef HbrVertex<OsdVertex>   OsdHbrVertex;
typedef HbrFace<OsdVertex>     OsdHbrFace;
typedef HbrHalfedge<OsdVertex> OsdHbrHalfedge;

typedef FarMesh<OsdVertex>              OsdFarMesh;
typedef FarMeshFactory<OsdVertex>       OsdFarMeshFactory;
typedef FarSubdivisionTables<OsdVertex> OsdFarMeshSubdivision;

//------------------------------------------------------------------------------
struct SimpleShape {
    std::string  name;
    Scheme       scheme;
    std::string  data;

    SimpleShape() { }
    SimpleShape( std::string const & idata, char const * iname, Scheme ischeme )
        : name(iname), scheme(ischeme), data(idata) { }
};

std::vector<SimpleShape> g_defaultShapes;

std::vector<float> g_orgPositions,
                   g_positions;

int g_currentShape = 0,
    g_level = 3,
    g_numElements = 3;

std::vector<int>   g_coarseEdges;
std::vector<float> g_coarseEdgeSharpness;
std::vector<float> g_coarseVertexSharpness;

int   g_running = 1,
      g_width = 1024,
      g_height = 1024,
      g_fullscreen = 0,
      g_drawCageEdges = 1,
      g_drawCageVertices = 0,
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

GLuint g_transformUB = 0,
       g_transformBinding = 0;

struct Transform {
    float ModelViewMatrix[16];
    float ProjectionMatrix[16];
    float ModelViewProjectionMatrix[16];
} g_transformData;


// performance
float g_evalTime = 0;
float g_computeTime = 0;
Stopwatch g_fpsTimer;

//------------------------------------------------------------------------------
static void
initializeShapes( ) {

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
    g_defaultShapes.push_back(SimpleShape(catmark_edgecorner ,"catmark_edgecorner", kCatmark));

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



}

//------------------------------------------------------------------------------
OsdCpuVertexBuffer * g_samplesVB=0;

int g_nsamples=1000,
    g_nsamplesFound=0;

GLuint g_cageEdgeVAO = 0,
       g_cageEdgeVBO = 0,
       g_cageVertexVAO = 0,
       g_cageVertexVBO = 0,
       g_samplesVAO=0;

GLhud g_hud;

//------------------------------------------------------------------------------
static void
createCoarseMesh( OsdHbrMesh * const hmesh, int nfaces ) {
    // save coarse topology (used for coarse mesh drawing)
    g_coarseEdges.clear();
    g_coarseEdgeSharpness.clear();
    g_coarseVertexSharpness.clear();

    for(int i=0; i<nfaces; ++i) {
        OsdHbrFace *face = hmesh->GetFace(i);
        int nv = face->GetNumVertices();
        for(int j=0; j<nv; ++j) {
            g_coarseEdges.push_back(face->GetVertex(j)->GetID());
            g_coarseEdges.push_back(face->GetVertex((j+1)%nv)->GetID());
            g_coarseEdgeSharpness.push_back(face->GetEdge(j)->GetSharpness());
        }
    }
    int nv = hmesh->GetNumVertices();
    for(int i=0; i<nv; ++i) {
        g_coarseVertexSharpness.push_back(hmesh->GetVertex(i)->GetSharpness());
    }
}

//------------------------------------------------------------------------------
static int
createRandomSamples( int nfaces, int nsamples, std::vector<OsdEvalCoords> & coords ) {

    coords.resize(nfaces * nsamples);

    OsdEvalCoords * coord = &coords[0];
    
    // large Pell prime number
    srand( static_cast<int>(2147483647) );
    
    for (int i=0; i<nfaces; ++i) {
        for (int j=0; j<nsamples; ++j) {
            coord->face = i;
            coord->u = (float)rand()/(float)RAND_MAX;
            coord->v = (float)rand()/(float)RAND_MAX;
            ++coord;
        }
    }
        
    return (int)coords.size();
}

//------------------------------------------------------------------------------
static int
getNumPtexFaces( OsdHbrMesh const * hmesh, int nfaces ) {

    OsdHbrFace * lastface = hmesh->GetFace( nfaces-1 );
    assert(lastface);
    
    int result = lastface->GetPtexIndex();
    
    result += (hmesh->GetSubdivision()->FaceIsExtraordinary(hmesh, lastface) ? 
                  lastface->GetNumVertices() : 0);

    return ++result;
}

//------------------------------------------------------------------------------
OsdCpuComputeContext * g_computeCtx = 0;

OsdCpuComputeController g_computeCtrl;

OsdCpuEvalLimitContext * g_evalCtx = 0;

OsdCpuEvalLimitController g_evalCtrl;

OsdVertexBufferDescriptor g_idesc( /*offset*/ 0, /*legnth*/ 3, /*stride*/ 3 ), 
                          g_odesc( /*offset*/ 0, /*legnth*/ 3, /*stride*/ 6 );

std::vector<OsdEvalCoords> g_coords;

OsdCpuGLVertexBuffer * g_Q=0,
                     * g_dQu=0,
                     * g_dQv=0;
                     
OsdFarMesh * g_fmesh=0;

//------------------------------------------------------------------------------
static void
updateGeom() {

    int nverts = (int)g_orgPositions.size() / 3;

    const float *p = &g_orgPositions[0];

    float r = sin(g_frame*0.001f) * g_moveScale;

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
    
    g_samplesVB->UpdateData( &g_positions[0], 0, nverts);
    
    g_computeCtrl.Refine( g_computeCtx, g_fmesh->GetKernelBatches(), g_samplesVB );

    s.Stop();
    g_computeTime = float(s.GetElapsed() * 1000.0f);


    // Run Eval pass to get the samples locations ------------------------------

    s.Start();
    
    // Reset the output buffer
    float * cpubuff = g_Q->BindCpuBuffer();
    memset( cpubuff, 0, g_Q->GetNumVertices()*g_Q->GetNumElements()*sizeof(float) );

    g_nsamplesFound=0;

    // Bind/Unbind of the vertex buffers to the context needs to happen 
    // outside of the parallel loop
    g_evalCtx->BindVertexBuffers( g_idesc, g_samplesVB, g_odesc, g_Q, g_dQu, g_dQv );

#ifdef OPENSUBDIV_HAS_OPENMP
    #pragma omp parallel for
#endif
    for (int i=0; i<(int)g_coords.size(); ++i) {
    
        int n = g_evalCtrl.EvalLimitSample<OsdCpuVertexBuffer,OsdCpuGLVertexBuffer>( g_coords[i], g_evalCtx, i );

        if (n) {

            // point colors
            float * color = cpubuff + i * 6 + 3;

            color[0] = g_coords[i].u;
            color[1] = 0.0f;
            color[2] = g_coords[i].v;
            
#ifdef OPENSUBDIV_HAS_OPENMP
            #pragma omp atomic
#endif
            g_nsamplesFound += n;
        }
    }
    
    g_evalCtx->UnbindVertexBuffers();
    
    g_Q->BindVBO();

    s.Stop();
    
    g_evalTime = float(s.GetElapsed() * 1000.0f);
}

//------------------------------------------------------------------------------
static void
createOsdMesh( const std::string &shape, int level, Scheme scheme=kCatmark ) {

    // Create HBR mesh
    OsdHbrMesh * hmesh = simpleHbr<OsdVertex>(shape.c_str(), scheme, g_orgPositions);

    g_positions.resize(g_orgPositions.size(),0.0f);

    int nfaces = hmesh->GetNumFaces(),
        nptexfaces = getNumPtexFaces(hmesh, nfaces);
    
    // Generate sample locations 
    int nsamples = createRandomSamples( nptexfaces, g_nsamples, g_coords );

    createCoarseMesh(hmesh, nfaces);



    // Create FAR mesh
    OsdFarMeshFactory factory( hmesh, level, /*adaptive*/ true);    
    
    delete g_fmesh;
    g_fmesh = factory.Create(/*fvar*/ false);
    
    int nverts = g_fmesh->GetNumVertices();
    

    
    // Create v-buffer & populate w/ positions
    delete g_samplesVB;
    g_samplesVB = OsdCpuVertexBuffer::Create(3, nverts);


        
    // Create a Compute context, used to "pose" the vertices
    delete g_computeCtx;
    g_computeCtx = OsdCpuComputeContext::Create(g_fmesh);
    
    g_computeCtrl.Refine( g_computeCtx, g_fmesh->GetKernelBatches(), g_samplesVB );
    

    
    // Create eval context & data buffers
    delete g_evalCtx;
    g_evalCtx = OsdCpuEvalLimitContext::Create(g_fmesh);

    delete g_Q;
    g_Q = OsdCpuGLVertexBuffer::Create(6,nsamples);

    delete g_dQu;
    g_dQu = OsdCpuGLVertexBuffer::Create(6,nsamples);

    delete g_dQv;
    g_dQv = OsdCpuGLVertexBuffer::Create(6,nsamples);
        
    updateGeom();

    // Bind g_Q as a GL_POINTS VBO
    glBindVertexArray(g_samplesVAO);
    
    glBindBuffer(GL_ARRAY_BUFFER, g_Q->BindVBO());

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 6, 0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 6, (float*)12);

    glBindVertexArray(0);
}

//------------------------------------------------------------------------------
struct Program
{
    GLuint program;
    GLuint uniformModelViewProjectionMatrix;
    GLuint attrPosition;
    GLuint attrColor;
} g_defaultProgram;

//------------------------------------------------------------------------------
static void
checkGLErrors(std::string const & where = "")
{
    GLuint err;
    while ((err = glGetError()) != GL_NO_ERROR) {

        std::cerr << "GL error: "
                  << (where.empty() ? "" : where + " ")
                  << err << "\n";
    }
}

//------------------------------------------------------------------------------
static GLuint
compileShader(GLenum shaderType, const char *source)
{
    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    checkGLErrors("compileShader");
    return shader;
}

//------------------------------------------------------------------------------
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
static inline void
setSharpnessColor(float s, float *r, float *g, float *b)
{
    //  0.0       2.0       4.0
    // green --- yellow --- red
    *r = std::min(1.0f, s * 0.5f);
    *g = std::min(1.0f, 2.0f - s*0.5f);
    *b = 0;
}

//------------------------------------------------------------------------------
static void
drawCageEdges() {

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
                          3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 6, 0);
    glVertexAttribPointer(g_defaultProgram.attrColor,
                          3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 6, (void*)12);

    glDrawArrays(GL_LINES, 0, (int)g_coarseEdges.size());

    glBindVertexArray(0);
    glUseProgram(0);
} 

//------------------------------------------------------------------------------
static void
drawCageVertices() {

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
                          3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 6, 0);
    glVertexAttribPointer(g_defaultProgram.attrColor,
                          3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 6, (void*)12);

    glPointSize(10.0f);
    glDrawArrays(GL_POINTS, 0, numPoints);
    glPointSize(1.0f);

    glBindVertexArray(0);
    glUseProgram(0);
}

//------------------------------------------------------------------------------
static void
drawSamples() {

    glUseProgram(g_defaultProgram.program);

    glUniformMatrix4fv(g_defaultProgram.uniformModelViewProjectionMatrix,
                       1, GL_FALSE, g_transformData.ModelViewProjectionMatrix);


    glBindVertexArray(g_samplesVAO);

    glPointSize(1.0f);
    glDrawArrays( GL_POINTS, 0,  (int)g_coords.size() );
    glPointSize(1.0f);

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

    s.Stop();
    float drawCpuTime = float(s.GetElapsed() * 1000.0f);
    s.Start();
    glFinish();
    s.Stop();
    float drawGpuTime = float(s.GetElapsed() * 1000.0f);

    drawSamples();

    if (g_drawCageEdges)
        drawCageEdges();
    
    if (g_drawCageVertices)
        drawCageVertices();

    if (g_hud.IsVisible()) {
        g_fpsTimer.Stop();
        double fps = 1.0/g_fpsTimer.GetElapsed();
        g_fpsTimer.Start();

        g_hud.DrawString(10, -120, "# Samples  : (%d/%d)", g_nsamplesFound, g_Q->GetNumVertices());
        g_hud.DrawString(10, -100, "Compute    : %.3f ms", g_computeTime);
        g_hud.DrawString(10, -80,  "Eval       : %.3f ms", g_evalTime);
        g_hud.DrawString(10, -60,  "GPU Draw   : %.3f ms", drawGpuTime);
        g_hud.DrawString(10, -40,  "CPU Draw   : %.3f ms", drawCpuTime);
        g_hud.DrawString(10, -20,  "FPS        : %3.1f", fps);

        g_hud.Flush();
    }

    glFinish();

    checkGLErrors("display leave");
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
#if GLFW_VERSION_MAJOR>=3
motion(GLFWwindow *, double dx, double dy) {
    int x=(int)dx, y=(int)dy;
#else
motion(int x, int y) {
#endif

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
#if GLFW_VERSION_MAJOR>=3
mouse(GLFWwindow *, int button, int state) {
#else
mouse(int button, int state) {
#endif

    if (button == 0 && state == GLFW_PRESS && g_hud.MouseClick(g_prev_x, g_prev_y))
        return;

    if (button < 3) {
        g_mbutton[button] = (state == GLFW_PRESS);
    }
}

//------------------------------------------------------------------------------
static void
#if GLFW_VERSION_MAJOR>=3
reshape(GLFWwindow *, int width, int height) {
#else
reshape(int width, int height) {
#endif

    g_width = width;
    g_height = height;
    
    g_hud.Rebuild(width, height);
}

//------------------------------------------------------------------------------
#if GLFW_VERSION_MAJOR>=3
void windowClose(GLFWwindow*) {
    g_running = false;
}
#else
int windowClose() {
    g_running = false;
    return GL_TRUE;
}
#endif

//------------------------------------------------------------------------------
static void
setSamples(bool add)
{
    g_nsamples += add ? 1000 : -1000;

    g_nsamples = std::max(0, g_nsamples);
    
    createOsdMesh( g_defaultShapes[g_currentShape].data, g_level, g_defaultShapes[ g_currentShape ].scheme );
}

//------------------------------------------------------------------------------
static void
#if GLFW_VERSION_MAJOR>=3
keyboard(GLFWwindow *, int key, int event) {
#else
keyboard(int key, int event) {
#endif

    if (event == GLFW_RELEASE) return;
    if (g_hud.KeyDown(tolower(key))) return;

    switch (key) {
        case 'Q': g_running = 0; break;
        
        case '=': setSamples(true); break;
        
        case '-': setSamples(false); break;
        
        case GLFW_KEY_ESC: g_hud.SetVisible(!g_hud.IsVisible()); break;
    }
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
callbackModel(int m)
{
    if (m < 0)
        m = 0;

    if (m >= (int)g_defaultShapes.size())
        m = (int)g_defaultShapes.size() - 1;

    g_currentShape = m;

    createOsdMesh( g_defaultShapes[m].data, g_level, g_defaultShapes[ g_currentShape ].scheme );
}

//------------------------------------------------------------------------------
static void
callbackLevel(int l)
{
    g_level = l;
    createOsdMesh( g_defaultShapes[g_currentShape].data, g_level, g_defaultShapes[ g_currentShape ].scheme );
}

//------------------------------------------------------------------------------
static void
callbackAnimate(bool checked, int m)
{
    g_moveScale = checked;
}

//------------------------------------------------------------------------------
static void
callbackFreeze(bool checked, int f)
{
    g_freeze = checked;
}

//------------------------------------------------------------------------------
static void
initHUD()
{
    g_hud.Init(g_width, g_height);

    g_hud.AddCheckBox("Animate vertices (M)", g_moveScale != 0, 350, 20, callbackAnimate, 0, 'm');
    g_hud.AddCheckBox("Freeze (spc)", false, 350, 40, callbackFreeze, 0, ' ');

    for (int i = 1; i < 11; ++i) {
        char level[16];
        sprintf(level, "Lv. %d", i);
        g_hud.AddRadioButton(3, level, i==g_level, 10, 170+i*20, callbackLevel, i, '0'+(i%10));
    }

    for (int i = 0; i < (int)g_defaultShapes.size(); ++i) {
        g_hud.AddRadioButton(4, g_defaultShapes[i].name.c_str(), i==0, -220, 10+i*16, callbackModel, i, 'n');
    }
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

    glGenVertexArrays(1, &g_cageVertexVAO);
    glGenVertexArrays(1, &g_cageEdgeVAO);
    glGenVertexArrays(1, &g_samplesVAO);
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
    glDeleteVertexArrays(1, &g_samplesVAO);
}

//------------------------------------------------------------------------------
static void
setGLCoreProfile()
{
#if GLFW_VERSION_MAJOR>=3
    #define glfwOpenWindowHint glfwWindowHint
    #define GLFW_OPENGL_VERSION_MAJOR GLFW_CONTEXT_VERSION_MAJOR
    #define GLFW_OPENGL_VERSION_MINOR GLFW_CONTEXT_VERSION_MINOR
#endif

    glfwOpenWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if not defined(__APPLE__)
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 4);
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
    glfwOpenWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#else
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3);
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
#endif
    
}

//------------------------------------------------------------------------------
int main(int, char**) {

    bool fullscreen = false;

    OsdSetErrorCallback(callbackError);
    
    initializeShapes();

    if (not glfwInit()) {
        printf("Failed to initialize GLFW\n");
        return 1;
    }

    static const char windowTitle[] = "OpenSubdiv evalViewer";
    
#define CORE_PROFILE
#ifdef CORE_PROFILE
    setGLCoreProfile();
#endif
    
#if GLFW_VERSION_MAJOR>=3
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
            GLFWvidmode vidmode = glfwGetVideoMode(g_primary);
            g_width = vidmode.width;
            g_height = vidmode.height;
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
    
#if not defined(__APPLE__)
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

    //std::string & data = g_defaultShapes[ g_currentShape ].data;
    //Scheme scheme = g_defaultShapes[ g_currentShape ].scheme;

    //createOsdMesh( data, g_level, scheme );

    initGL();
    linkDefaultProgram();
    
    glfwSwapInterval(0);

    initHUD();
    callbackModel(g_currentShape);

    while (g_running) {
        idle();
        display();
        
#if GLFW_VERSION_MAJOR>=3
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
