//
//     Copyright 2013 Pixar
//
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License
//     and the following modification to it: Section 6 Trademarks.
//     deleted and replaced with:
//
//     6. Trademarks. This License does not grant permission to use the
//     trade names, trademarks, service marks, or product names of the
//     Licensor and its affiliates, except as required for reproducing
//     the content of the NOTICE file.
//
//     You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//     Unless required by applicable law or agreed to in writing,
//     software distributed under the License is distributed on an
//     "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
//     either express or implied.  See the License for the specific
//     language governing permissions and limitations under the
//     License.
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
    GLFWwindow* g_window=0;
    GLFWmonitor* g_primary=0;
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
OpenSubdiv::OsdGLMeshInterface *g_mesh;

#include <common/shape_utils.h>
#include "../common/stopwatch.h"
#include "../common/simple_math.h"
#include "../common/gl_hud.h"

static const char *shaderSource =
#include "shader.inc"
;
static const char *paintShaderSource =
#include "paintShader.inc"
;

#include <cfloat>
#include <vector>
#include <fstream>
#include <sstream>

typedef OpenSubdiv::HbrMesh<OpenSubdiv::OsdVertex>     OsdHbrMesh;
typedef OpenSubdiv::HbrVertex<OpenSubdiv::OsdVertex>   OsdHbrVertex;
typedef OpenSubdiv::HbrFace<OpenSubdiv::OsdVertex>     OsdHbrFace;
typedef OpenSubdiv::HbrHalfedge<OpenSubdiv::OsdVertex> OsdHbrHalfedge;

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

int g_level = 2;
int g_tessLevel = 6;

std::vector<float> g_orgPositions;

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
    float ModelViewInverseMatrix[16];
    float ProjectionWithoutPickMatrix[16];
} g_transformData;

GLuint g_primQuery = 0;
GLuint g_vao = 0;

GLuint g_paintTexture = 0;

GLuint g_depthTexture = 0;

int g_running = 1,
    g_wire = 2,
    g_displayColor = 1,
    g_displayDisplacement = 0,
    g_mbutton[3] = {0, 0, 0};

int g_brushSize = 500;
int g_frame = 0;

GLuint g_ptexPages = 0,
    g_ptexLayouts = 0,
    g_ptexTexels = 0;

int g_pageSize = 512;

struct SimpleShape {
    std::string  name;
    Scheme       scheme;
    std::string  data;

    SimpleShape() { }
    SimpleShape( std::string const & idata, char const * iname, Scheme ischeme )
        : name(iname), scheme(ischeme), data(idata) { }
};

std::vector<SimpleShape> g_defaultShapes;

int g_currentShape = 0;

#define NUM_FPS_TIME_SAMPLES 6
float g_fpsTimeSamples[NUM_FPS_TIME_SAMPLES] = {0,0,0,0,0,0};
int   g_currentFpsTimeSample = 0;
Stopwatch g_fpsTimer;

static void
checkGLErrors(std::string const & where = "")
{
    GLuint err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        /*
        std::cerr << "GL error: "
                  << (where.empty() ? "" : where + " ")
                  << err << "\n";
        */
    }
}

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

}

//------------------------------------------------------------------------------
static void
updateGeom() {

    int nverts = (int)g_orgPositions.size() / 3;

    std::vector<float> vertex;
    vertex.reserve(nverts*3);

    const float *p = &g_orgPositions[0];

    g_frame++;

//    float r = sin(frame*0.01f); // * g_moveScale;
    float r = 0;
    for (int i = 0; i < nverts; ++i) {
        //float move = 0.05f*cosf(p[0]*20+g_frame*0.01f);
        float ct = cos(p[2] * r);
        float st = sin(p[2] * r);
        vertex.push_back(p[0]*ct + p[1]*st);
        vertex.push_back(-p[0]*st + p[1]*ct);
        vertex.push_back(p[2]);
        p += 3;
    }

    g_mesh->UpdateVertexBuffer(&vertex[0], 0, nverts);
//    g_mesh->UpdateVertexBuffer(&g_orgPositions[0], 0, nverts);
    g_mesh->Refine();
    g_mesh->Synchronize();
}

//------------------------------------------------------------------------------
static GLuint
genTextureBuffer(GLenum format, GLsizeiptr size, GLvoid const * data) {

    GLuint buffer, result;
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_TEXTURE_BUFFER, buffer);
    glBufferData(GL_TEXTURE_BUFFER, size, data, GL_STATIC_DRAW);

    glGenTextures(1, & result);
    glBindTexture(GL_TEXTURE_BUFFER, result);
    glTexBuffer(GL_TEXTURE_BUFFER, format, buffer);

    // need to reset texture binding before deleting the source buffer.
    glBindTexture(GL_TEXTURE_BUFFER, 0);
    glDeleteBuffers(1, &buffer);

    return result;
}

static void
createOsdMesh() {
    const char *shape = g_defaultShapes[g_currentShape].data.c_str();
    int level = g_level;
    Scheme scheme = g_defaultShapes[g_currentShape].scheme;

    checkGLErrors("create osd enter");
    // generate Hbr representation from "obj" description
    OsdHbrMesh * hmesh = simpleHbr<OpenSubdiv::OsdVertex>(shape, scheme, g_orgPositions);

    // count ptex face id
    int numPtexFace = 0;
    int numFace = hmesh->GetNumFaces();
    for (int i = 0; i < numFace; ++i) {
        numPtexFace = std::max(numPtexFace, hmesh->GetFace(i)->GetPtexIndex());
    }
    numPtexFace++;

    delete g_mesh;
    g_mesh = NULL;

    bool doAdaptive = true;
    OpenSubdiv::OsdMeshBitset bits;
    bits.set(OpenSubdiv::MeshAdaptive, doAdaptive);
    bits.set(OpenSubdiv::MeshPtexData, true);

    if (not g_cpuComputeController) {
        g_cpuComputeController = new OpenSubdiv::OsdCpuComputeController();
    }
    g_mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdCpuGLVertexBuffer,
        OpenSubdiv::OsdCpuComputeController,
        OpenSubdiv::OsdGLDrawContext>(
            g_cpuComputeController,
            hmesh, 3, 0, level, bits);

    // Hbr mesh can be deleted
    delete hmesh;

    // compute model bounding
    float min[3] = { FLT_MAX,  FLT_MAX,  FLT_MAX};
    float max[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (size_t i=0; i < g_orgPositions.size()/3; ++i) {
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

    updateGeom();

    // -------- VAO 
    glBindVertexArray(g_vao);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_mesh->GetDrawContext()->GetPatchIndexBuffer());
    glBindBuffer(GL_ARRAY_BUFFER, g_mesh->BindVertexBuffer());

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 3, 0);

    glBindVertexArray(0);

    // -------- create ptex
    if (g_ptexPages) glDeleteTextures(1, &g_ptexPages);
    if (g_ptexLayouts) glDeleteTextures(1, &g_ptexLayouts);
    if (g_ptexTexels) glDeleteTextures(1, &g_ptexTexels);

    std::vector<int> pages;
    std::vector<float> layouts;
    for (int i = 0; i < numPtexFace; ++i) {
        pages.push_back(i);
        layouts.push_back(0);
        layouts.push_back(0);
        layouts.push_back(1);
        layouts.push_back(1);
    }
    g_ptexPages = genTextureBuffer(GL_R32I,
                                   numPtexFace * sizeof(GLint), &pages[0]);
    
    g_ptexLayouts = genTextureBuffer(GL_RGBA32F,
                                     numPtexFace * 4 * sizeof(GLfloat),
                                     &layouts[0]);

    // actual texels texture array
    glGenTextures(1, &g_ptexTexels);
    glBindTexture(GL_TEXTURE_2D_ARRAY, g_ptexTexels);

    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    g_pageSize = std::min(512, (int)sqrt((float)1024*1024*1024/64/numPtexFace));

    int pageSize = g_pageSize;

    std::vector<float> texels;
    texels.resize(pageSize*pageSize*numPtexFace);
    // allocate ptex
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_R32F, 
                 pageSize, pageSize, numPtexFace, 0, GL_RED, GL_FLOAT, &texels[0]);

    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
}

//------------------------------------------------------------------------------
static void
fitFrame() {

    g_pan[0] = g_pan[1] = 0;
    g_dolly = g_size;
}

//------------------------------------------------------------------------------
union Effect {
    struct {
        int color:1;
        int displacement:1;
        int paint:1;
        unsigned int wire:2;
    };
    int value;

    bool operator < (const Effect &e) const {
        return value < e.value;
    }
};

typedef std::pair<OpenSubdiv::OsdDrawContext::PatchDescriptor,Effect> EffectDesc;

class EffectDrawRegistry : public OpenSubdiv::OsdGLDrawRegistry<EffectDesc> {

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

    sconfig->commonShader.AddDefine("USE_PTEX_COORD");
    sconfig->commonShader.AddDefine("OSD_USER_TRANSFORM_UNIFORMS", "mat4 ProjectionWithoutPickMatrix;");

    if (effect.color) {
        sconfig->commonShader.AddDefine("USE_PTEX_COLOR");
    }
    if (effect.displacement) {
        sconfig->commonShader.AddDefine("USE_PTEX_DISPLACEMENT");
    }
    sconfig->commonShader.AddDefine("OSD_ENABLE_PATCH_CULL");
    sconfig->commonShader.AddDefine("OSD_ENABLE_SCREENSPACE_TESSELLATION");

    const char *glslVersion = "#version 420\n";

    if (effect.paint) {
        sconfig->vertexShader.version = glslVersion;
        sconfig->geometryShader.version = glslVersion;
        sconfig->geometryShader.source = paintShaderSource;
        sconfig->geometryShader.AddDefine("GEOMETRY_SHADER");
        sconfig->fragmentShader.version = glslVersion;
        sconfig->fragmentShader.source = paintShaderSource;
        sconfig->fragmentShader.AddDefine("FRAGMENT_SHADER");
        return sconfig;
    }

    sconfig->geometryShader.AddDefine("SMOOTH_NORMALS");
    sconfig->geometryShader.source = shaderSource;
    sconfig->geometryShader.version = glslVersion;
    sconfig->geometryShader.AddDefine("GEOMETRY_SHADER");

    sconfig->fragmentShader.source = shaderSource;
    sconfig->fragmentShader.version = glslVersion;
    sconfig->fragmentShader.AddDefine("FRAGMENT_SHADER");

    sconfig->geometryShader.AddDefine("PRIM_TRI");
    sconfig->fragmentShader.AddDefine("PRIM_TRI");

    if (effect.wire == 0) {
        sconfig->geometryShader.AddDefine("GEOMETRY_OUT_WIRE");
        sconfig->fragmentShader.AddDefine("GEOMETRY_OUT_WIRE");
    } else if (effect.wire == 1) {
        sconfig->geometryShader.AddDefine("GEOMETRY_OUT_FILL");
        sconfig->fragmentShader.AddDefine("GEOMETRY_OUT_FILL");
    } else if (effect.wire == 2) {
        sconfig->geometryShader.AddDefine("GEOMETRY_OUT_LINE");
        sconfig->fragmentShader.AddDefine("GEOMETRY_OUT_LINE");
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

    g_lightingBinding = 2;
    uboIndex = glGetUniformBlockIndex(config->program, "Lighting");
    if (uboIndex != GL_INVALID_INDEX)
        glUniformBlockBinding(config->program, uboIndex, g_lightingBinding);

    GLint loc;
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

    return config;
}

EffectDrawRegistry effectRegistry;

//------------------------------------------------------------------------------
static GLuint
bindProgram(Effect effect, OpenSubdiv::OsdDrawContext::PatchArray const & patch)
{
    EffectDesc effectDesc(patch.GetDescriptor(), effect);
    EffectDrawRegistry::ConfigType *
        config = effectRegistry.GetDrawConfig(effectDesc);

    GLuint program = config->program;

    glUseProgram(program);

    if (effect.paint) {
        // set image
        GLint texImage = glGetUniformLocation(program, "outTextureImage");
        glUniform1i(texImage, 0);
        glBindImageTexture(0, g_ptexTexels, 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32F);

        GLint paintTexture = glGetUniformLocation(program, "paintTexture");
        glUniform1i(paintTexture, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, g_paintTexture);

        GLint depthTexture = glGetUniformLocation(program, "depthTexture");
        glUniform1i(depthTexture, 1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, g_depthTexture);

        GLint imageSize = glGetUniformLocation(program, "imageSize");
        glUniform1i(imageSize, g_pageSize);

        glActiveTexture(GL_TEXTURE0);
    }

    // color ptex
    GLint texData = glGetUniformLocation(program, "textureImage_Data");
    GLint texPacking = glGetUniformLocation(program, "textureImage_Packing");
    GLint texPages = glGetUniformLocation(program, "textureImage_Pages");
    
    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D_ARRAY, g_ptexTexels);
    glProgramUniform1i(program, texData, 5);
    
    glActiveTexture(GL_TEXTURE6);
    glBindTexture(GL_TEXTURE_BUFFER, g_ptexLayouts);
    glProgramUniform1i(program, texPacking, 6);
    
    glActiveTexture(GL_TEXTURE7);
    glBindTexture(GL_TEXTURE_BUFFER, g_ptexPages);
    glProgramUniform1i(program, texPages, 7);
    
    glActiveTexture(GL_TEXTURE0);

    return program;
}

//------------------------------------------------------------------------------
static void
display() {

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glViewport(0, 0, g_width, g_height);

    // primitive counting
    glBeginQuery(GL_PRIMITIVES_GENERATED, g_primQuery);

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


    glActiveTexture(GL_TEXTURE0);

    // make sure that the vertex buffer is interoped back as a GL resources.
    g_mesh->BindVertexBuffer();
    
    glBindVertexArray(g_vao);

    OpenSubdiv::OsdDrawContext::PatchArrayVector const & patches = g_mesh->GetDrawContext()->patchArrays;

    // patch drawing
    for (int i=0; i<(int)patches.size(); ++i) {
        OpenSubdiv::OsdDrawContext::PatchArray const & patch = patches[i];
        OpenSubdiv::OsdDrawContext::PatchDescriptor desc = patch.GetDescriptor();

        GLenum primType = GL_PATCHES;
        glPatchParameteri(GL_PATCH_VERTICES, desc.GetNumControlVertices());

        Effect effect;
        effect.color = g_displayColor;
        effect.displacement = g_displayDisplacement;
        effect.wire = g_wire;
        effect.paint = 0;

        GLuint program = bindProgram(effect, patch);
        GLuint diffuseColor = glGetUniformLocation(program, "diffuseColor");
        glProgramUniform4f(program, diffuseColor, 1, 1, 1, 1);

        GLuint uniformGregoryQuadOffsetBase =
	  glGetUniformLocation(program, "OsdGregoryQuadOffsetBase");
        GLuint uniformPrimitiveIdBase =
	  glGetUniformLocation(program, "OsdPrimitiveIdBase");
        glProgramUniform1i(program, uniformGregoryQuadOffsetBase,
			   patch.GetQuadOffsetIndex());
        glProgramUniform1i(program, uniformPrimitiveIdBase,
			   patch.GetPatchIndex());

        if (g_wire == 0) {
            glDisable(GL_CULL_FACE);
        }
        glDrawElements(primType,
                       patch.GetNumIndices(), GL_UNSIGNED_INT,
                       (void *)(patch.GetVertIndex() * sizeof(unsigned int)));
        if (g_wire == 0) {
            glEnable(GL_CULL_FACE);
        }
    }

    glBindVertexArray(0);
    glUseProgram(0);

    glEndQuery(GL_PRIMITIVES_GENERATED);

    glBindTexture(GL_TEXTURE_2D, g_depthTexture);
    glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, g_width, g_height);
    glBindTexture(GL_TEXTURE_2D, 0);

    GLuint numPrimsGenerated = 0;
    glGetQueryObjectuiv(g_primQuery, GL_QUERY_RESULT, &numPrimsGenerated);

    if (g_hud.IsVisible()) {
        g_fpsTimer.Stop();
        double fps = 1.0/g_fpsTimer.GetElapsed();
        g_fpsTimer.Start();
        // Avereage fps over a defined number of time samples for
        // easier reading in the HUD
        g_fpsTimeSamples[g_currentFpsTimeSample++] = float(fps);
        if (g_currentFpsTimeSample >= NUM_FPS_TIME_SAMPLES)
            g_currentFpsTimeSample = 0;
        double averageFps = 0;
        for (int i=0; i< NUM_FPS_TIME_SAMPLES; ++i) {
            averageFps += g_fpsTimeSamples[i]/(float)NUM_FPS_TIME_SAMPLES;
        }

        g_hud.DrawString(10, -180, "Tess level (+/-): %d", g_tessLevel);
        if (numPrimsGenerated > 1000000) {
            g_hud.DrawString(10, -160, "Primitives      : %3.1f million", (float)numPrimsGenerated/1000000.0);
        } else if (numPrimsGenerated > 1000) {
            g_hud.DrawString(10, -160, "Primitives      : %3.1f thousand", (float)numPrimsGenerated/1000.0);
        } else {
            g_hud.DrawString(10, -160, "Primitives      : %d", numPrimsGenerated);
        }
        g_hud.DrawString(10, -140, "Vertices        : %d", g_mesh->GetNumVertices());
        g_hud.DrawString(10, -20,  "FPS             : %3.1f", averageFps);
    }

    g_hud.Flush();

    glFinish();

    checkGLErrors("display leave");

#if GLFW_VERSION_MAJOR>=3
    glfwSwapBuffers(g_window);
#else
    glfwSwapBuffers();
#endif

}

void
drawStroke(int x, int y) 
{
    glViewport(0, 0, g_pageSize, g_pageSize);

    // prepare view matrix
    double aspect = g_width/(double)g_height;
    int viewport[4] = {0, 0, g_width, g_height};
    float pick[16], pers[16];
    perspective(pers, 45.0f, (float)aspect, 0.01f, 500.0f);
    pickMatrix(pick, (float)x, (float)g_height-y, g_brushSize*0.5f, g_brushSize*0.5f, viewport);
    multMatrix(g_transformData.ProjectionMatrix, pers, pick);
    multMatrix(g_transformData.ModelViewProjectionMatrix,
               g_transformData.ModelViewMatrix,
               g_transformData.ProjectionMatrix);
    memcpy(g_transformData.ProjectionWithoutPickMatrix, pers, sizeof(float)*16);
    
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

    tessellationData.TessLevel = static_cast<float>(1 << (g_tessLevel>>1));

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

    glActiveTexture(GL_TEXTURE0);

    // make sure that the vertex buffer is interoped back as a GL resources.
    g_mesh->BindVertexBuffer();
    
    glBindVertexArray(g_vao);

    OpenSubdiv::OsdDrawContext::PatchArrayVector const & patches = g_mesh->GetDrawContext()->patchArrays;

    // patch drawing
    for (int i=0; i<(int)patches.size(); ++i) {
        OpenSubdiv::OsdDrawContext::PatchArray const & patch = patches[i];
        OpenSubdiv::OsdDrawContext::PatchDescriptor desc = patch.GetDescriptor();

        GLenum primType = GL_PATCHES;
        glPatchParameteri(GL_PATCH_VERTICES, desc.GetNumControlVertices());

        Effect effect;
        effect.color = 0;
        effect.displacement = g_displayDisplacement;
        effect.wire = 1;
        effect.paint = 1;
        
        GLuint program = bindProgram(effect, patch);
        GLuint uniformGregoryQuadOffsetBase =
	  glGetUniformLocation(program, "OsdGregoryQuadOffsetBase");
        GLuint uniformPrimitiveIdBase =
	  glGetUniformLocation(program, "OsdPrimitiveIdBase");
        glProgramUniform1i(program, uniformGregoryQuadOffsetBase,
			   patch.GetQuadOffsetIndex());
        glProgramUniform1i(program, uniformPrimitiveIdBase,
			   patch.GetPatchIndex());

        glDrawElements(primType,
                       patch.GetNumIndices(), GL_UNSIGNED_INT,
                       (void *)(patch.GetVertIndex() * sizeof(unsigned int)));
    }

    glBindVertexArray(0);
    glUseProgram(0);

    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                    GL_TEXTURE_FETCH_BARRIER_BIT);

    checkGLErrors("display leave");
}

//------------------------------------------------------------------------------
static void
#if GLFW_VERSION_MAJOR>=3
motion(GLFWwindow * w, double dx, double dy) {
    int x=(int)dx, y=(int)dy;
#else
motion(int x, int y) {
#endif

#if GLFW_VERSION_MAJOR>=3
    if (glfwGetKey(w,GLFW_KEY_LEFT_ALT)) {
#else
    if (glfwGetKey(GLFW_KEY_LALT)) {
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
    } else {
        if (g_mbutton[0] && !g_mbutton[1] && !g_mbutton[2]) {
            // paint something into screen
            drawStroke(x, y);
        }
    }

    g_prev_x = x;
    g_prev_y = y;
}

//------------------------------------------------------------------------------
static void
#if GLFW_VERSION_MAJOR>=3
mouse(GLFWwindow * w, int button, int state, int mods) {
#else
mouse(int button, int state) {
#endif

    if (button == 0 && state == GLFW_PRESS && g_hud.MouseClick(g_prev_x, g_prev_y))
        return;

    if (button < 3) {
        g_mbutton[button] = (state == GLFW_PRESS);
    }

#if GLFW_VERSION_MAJOR>=3
    if (not glfwGetKey(w, GLFW_KEY_LEFT_ALT)) {
#else
    if (not glfwGetKey(GLFW_KEY_LALT)) {
#endif
        if (g_mbutton[0] && !g_mbutton[1] && !g_mbutton[2]) {
            drawStroke(g_prev_x, g_prev_y);
        }
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

    // prepare depth texture
    if (g_depthTexture == 0) glGenTextures(1, &g_depthTexture);
    glBindTexture(GL_TEXTURE_2D, g_depthTexture);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0,
                 GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

}

void reshape() {
#if GLFW_VERSION_MAJOR>=3
    reshape(g_window, g_width, g_height);
#else
    reshape(g_width, g_height);
#endif
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
toggleFullScreen() {
    // XXXX manuelk : to re-implement from glut
}

//------------------------------------------------------------------------------
static void
#if GLFW_VERSION_MAJOR>=3
keyboard(GLFWwindow *, int key, int scancode, int event, int mods) {
#else
#define GLFW_KEY_ESCAPE GLFW_KEY_ESC
keyboard(int key, int event) {
#endif

    if (event == GLFW_RELEASE) return;
    if (g_hud.KeyDown(tolower(key))) return;

    switch (key) {
        case 'Q': g_running = 0; break;
        case 'F': fitFrame(); break;
        case GLFW_KEY_TAB: toggleFullScreen(); break;
        case '+':
        case '=':  g_tessLevel++; break;
        case '-':  g_tessLevel = std::max(1, g_tessLevel-1); break;
        case GLFW_KEY_ESCAPE: g_hud.SetVisible(!g_hud.IsVisible()); break;
    }
}

//------------------------------------------------------------------------------
static void
callbackWireframe(int b)
{
    g_wire = b;
}

static void
callbackDisplay(bool checked, int n)
{
    if (n == 0) g_displayColor = !g_displayColor;
    else if (n == 1) g_displayDisplacement = !g_displayDisplacement;
}

static void
callbackLevel(int l)
{
    g_level = l;
    createOsdMesh();
}

static void
callbackModel(int m)
{
    if (m < 0)
        m = 0;

    if (m >= (int)g_defaultShapes.size())
        m = (int)g_defaultShapes.size() - 1;

    g_currentShape = m;
    createOsdMesh();
}

static void
initHUD()
{
    g_hud.Init(g_width, g_height);

    g_hud.AddRadioButton(1, "Wire (W)",    g_wire == 0,  200, 10, callbackWireframe, 0, 'w');
    g_hud.AddRadioButton(1, "Shaded",      g_wire == 1, 200, 30, callbackWireframe, 1, 'w');
    g_hud.AddRadioButton(1, "Wire+Shaded", g_wire == 2, 200, 50, callbackWireframe, 2, 'w');

    g_hud.AddCheckBox("Color (C)",  g_displayColor != 0, 350, 10, callbackDisplay, 0, 'c');
    g_hud.AddCheckBox("Displacement (D)",  g_displayDisplacement != 0, 350, 30, callbackDisplay, 1, 'd');

    for (int i = 1; i < 11; ++i) {
        char level[16];
        sprintf(level, "Lv. %d", i);
        g_hud.AddRadioButton(3, level, i==2, 10, 170+i*20, callbackLevel, i, '0'+(i%10));
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

    glGenVertexArrays(1, &g_vao);
    glGenTextures(1, &g_paintTexture);

    static const GLfloat border[] = { 0.0, 0.0, 0.0, 0.0 };

    // create brush-size buffer
    glBindTexture(GL_TEXTURE_2D, g_paintTexture);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border);

    int reso = 64;

    std::vector<float> values;
    for(int yy = 0; yy < reso; ++yy) {
        for (int xx = 0; xx < reso; ++xx) {
            float r = sqrtf((xx-reso*0.5f)*(xx-reso*0.5f)+
                            (yy-reso*0.5f)*(yy-reso*0.5f))/(reso*0.5f);
            float v = 0.5f*std::max(0.0f, expf(-r*r)-0.4f);
            values.push_back(v);
            values.push_back(v);
            values.push_back(v);
        }
    }

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, reso, reso, 0, GL_RGB, GL_FLOAT, &values[0]);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenQueries(1, &g_primQuery);
}

//------------------------------------------------------------------------------
static void
uninitGL() {

    if (g_vao) glDeleteVertexArrays(1, &g_vao);
    if (g_paintTexture) glDeleteTextures(1, &g_paintTexture);
    if (g_depthTexture) glDeleteTextures(1, &g_depthTexture);
    if (g_primQuery) glDeleteQueries(1, &g_primQuery);
    if (g_ptexPages) glDeleteTextures(1, &g_ptexPages);
    if (g_ptexLayouts) glDeleteTextures(1, &g_ptexLayouts);
    if (g_ptexTexels) glDeleteTextures(1, &g_ptexTexels);

    if (g_mesh)
        delete g_mesh;

    delete g_cpuComputeController;
}

//------------------------------------------------------------------------------
static void
idle() {
    updateGeom();
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

int main(int argc, char ** argv)
{
    bool fullscreen = false;
    std::string str;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-d"))
            g_level = atoi(argv[++i]);
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

    static const char windowTitle[] = "OpenSubdiv painting test";
    
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

#if GLFW_VERSION_MAJOR>=3
    glfwSetWindowSizeCallback(g_window, reshape);
    // as of GLFW 3.0.1 this callback is not implicit
    reshape();
#else
    glfwSetWindowSizeCallback(reshape);
#endif

    glfwSwapInterval(0);

    initHUD();
    callbackModel(g_currentShape);

    while (g_running) {
        idle();
        display();

#if GLFW_VERSION_MAJOR>=3
        glfwPollEvents();
#endif
    }

    uninitGL();
    glfwTerminate();
}

//------------------------------------------------------------------------------
