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

#include <opensubdiv/far/error.h>
#include <opensubdiv/far/ptexIndices.h>
#include <opensubdiv/osd/cpuEvaluator.h>
#include <opensubdiv/osd/cpuGLVertexBuffer.h>
#include <opensubdiv/osd/glMesh.h>
OpenSubdiv::Osd::GLMeshInterface *g_mesh;

#include "../../regression/common/far_utils.h"
#include "../../regression/common/arg_utils.h"
#include "../common/viewerArgsUtils.h"
#include "../common/stopwatch.h"
#include "../common/simple_math.h"
#include "../common/glHud.h"
#include "../common/glShaderCache.h"
#include "../common/glUtils.h"

#include "init_shapes.h"

#include <opensubdiv/osd/glslPatchShaderSource.h>
static const char *shaderSource =
#include "shader.gen.h"
;
static const char *paintShaderSource =
#include "paintShader.gen.h"
;

#include <cfloat>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

float g_rotate[2] = {0, 0},
      g_dolly = 5,
      g_pan[2] = {0, 0},
      g_center[3] = {0, 0, 0},
      g_size = 0;

bool  g_yup = false;

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

int g_brushSize = 100;
int g_frame = 0;

GLuint g_ptexPages = 0,
    g_ptexLayouts = 0,
    g_ptexTexels = 0;

int g_pageSize = 512;

int g_currentShape = 0;

#define NUM_FPS_TIME_SAMPLES 6
float g_fpsTimeSamples[NUM_FPS_TIME_SAMPLES] = {0,0,0,0,0,0};
int   g_currentFpsTimeSample = 0;
Stopwatch g_fpsTimer;

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

    ShapeDesc const & shapeDesc = g_defaultShapes[g_currentShape];

    Shape * shape = Shape::parseObj(shapeDesc);

    checkGLErrors("create osd enter");

    g_orgPositions=shape->verts;

    // create Far mesh (topology)
    OpenSubdiv::Sdc::SchemeType sdctype = GetSdcType(*shape);
    OpenSubdiv::Sdc::Options sdcoptions = GetSdcOptions(*shape);

    // Recall this application should not accept or include Bilinear shapes
    assert(sdctype != OpenSubdiv::Sdc::SCHEME_BILINEAR);

    OpenSubdiv::Far::TopologyRefiner * refiner =
        OpenSubdiv::Far::TopologyRefinerFactory<Shape>::Create(*shape,
            OpenSubdiv::Far::TopologyRefinerFactory<Shape>::Options(sdctype, sdcoptions));

    // count ptex face id
    OpenSubdiv::Far::PtexIndices ptexIndices(*refiner);
    int numPtexFaces = ptexIndices.GetNumFaces();

    delete g_mesh;
    g_mesh = NULL;

    bool doAdaptive = true;
    OpenSubdiv::Osd::MeshBitset bits;
    bits.set(OpenSubdiv::Osd::MeshAdaptive, doAdaptive);
    bits.set(OpenSubdiv::Osd::MeshEndCapGregoryBasis, true);

    g_mesh = new OpenSubdiv::Osd::Mesh<OpenSubdiv::Osd::CpuGLVertexBuffer,
                                       OpenSubdiv::Far::StencilTable,
                                       OpenSubdiv::Osd::CpuEvaluator,
                                       OpenSubdiv::Osd::GLPatchTable>(
                                           refiner, 3, 0, g_level, bits);

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

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_mesh->GetPatchTable()->GetPatchIndexBuffer());
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
    for (int i = 0; i < numPtexFaces; ++i) {
        pages.push_back(i);
        layouts.push_back(0);
        layouts.push_back(0);
        layouts.push_back(1);
        layouts.push_back(1);
    }
    g_ptexPages = genTextureBuffer(GL_R32I,
                                   numPtexFaces * sizeof(GLint), &pages[0]);

    g_ptexLayouts = genTextureBuffer(GL_RGBA32F,
                                     numPtexFaces * 4 * sizeof(GLfloat),
                                     &layouts[0]);

    // actual texels texture array
    glGenTextures(1, &g_ptexTexels);
    glBindTexture(GL_TEXTURE_2D_ARRAY, g_ptexTexels);

    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    g_pageSize = std::min(512, (int)sqrt((float)1024*1024*1024/64/numPtexFaces));

    int pageSize = g_pageSize;

    std::vector<float> texels;
    texels.resize(pageSize*pageSize*numPtexFaces);
    // allocate ptex
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_R32F,
                 pageSize, pageSize, numPtexFaces, 0, GL_RED, GL_FLOAT, &texels[0]);

    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

    checkGLErrors("create osd exit");
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

struct EffectDesc {
    EffectDesc(OpenSubdiv::Far::PatchDescriptor desc,
               Effect effect) : desc(desc), effect(effect),
                                maxValence(0), numElements(0) { }

    OpenSubdiv::Far::PatchDescriptor desc;
    Effect effect;
    int maxValence;
    int numElements;

    bool operator < (const EffectDesc &e) const {
        return desc < e.desc || (desc == e.desc &&
              (maxValence < e.maxValence || ((maxValence == e.maxValence) &&
              (effect < e.effect))));
    }
};

// ---------------------------------------------------------------------------

class ShaderCache : public GLShaderCache<EffectDesc> {
public:
    virtual GLDrawConfig *CreateDrawConfig(EffectDesc const &effectDesc) {

        using namespace OpenSubdiv;

        // compile shader program
        const char *glslVersion = "#version 420\n";
        GLDrawConfig *config = new GLDrawConfig(glslVersion);

        Far::PatchDescriptor::Type type = effectDesc.desc.GetType();

        std::stringstream ss;
        if (effectDesc.effect.color) {
            ss << "#define USE_PTEX_COLOR\n";
        }
        if (effectDesc.effect.displacement) {
            ss << "#define USE_PTEX_DISPLACEMENT\n";
        }
        ss << "#define OSD_ENABLE_SCREENSPACE_TESSELLATION\n";
        if (effectDesc.effect.wire == 0) {
            ss << "#define GEOMETRY_OUT_WIRE\n";
        } else if (effectDesc.effect.wire == 1) {
            ss << "#define GEOMETRY_OUT_FILL\n";
        } else {
            ss << "#define GEOMETRY_OUT_LINE\n";
        }

        // for legacy gregory
        ss << "#define OSD_MAX_VALENCE " << effectDesc.maxValence << "\n";
        ss << "#define OSD_NUM_ELEMENTS " << effectDesc.numElements << "\n";

        // include osd PatchCommon
        ss << Osd::GLSLPatchShaderSource::GetCommonShaderSource();
        std::string common = ss.str();
        ss.str("");

        // vertex shader
        ss << common
           << (effectDesc.desc.IsAdaptive() ? "" : "#define VERTEX_SHADER\n")
           << (effectDesc.effect.paint ? paintShaderSource : shaderSource)
           << Osd::GLSLPatchShaderSource::GetVertexShaderSource(type);
        config->CompileAndAttachShader(GL_VERTEX_SHADER, ss.str());
        ss.str("");

        if (effectDesc.desc.IsAdaptive()) {
            // tess control shader
            ss << common
               << (effectDesc.effect.paint ? paintShaderSource : shaderSource)
               << Osd::GLSLPatchShaderSource::GetTessControlShaderSource(type);
            config->CompileAndAttachShader(GL_TESS_CONTROL_SHADER, ss.str());
            ss.str("");

            // tess eval shader
            ss << common
               << (effectDesc.effect.paint ? paintShaderSource : shaderSource)
               << Osd::GLSLPatchShaderSource::GetTessEvalShaderSource(type);
            config->CompileAndAttachShader(GL_TESS_EVALUATION_SHADER, ss.str());
            ss.str("");
        }

        // geometry shader
        ss << common
           << "#define GEOMETRY_SHADER\n" // for my shader source
           << (effectDesc.effect.paint ? paintShaderSource : shaderSource);
        config->CompileAndAttachShader(GL_GEOMETRY_SHADER, ss.str());
        ss.str("");

        // fragment shader
        ss << common
           << "#define FRAGMENT_SHADER\n" // for my shader source
           << (effectDesc.effect.paint ? paintShaderSource : shaderSource);
        config->CompileAndAttachShader(GL_FRAGMENT_SHADER, ss.str());
        ss.str("");

        if (!config->Link()) {
            delete config;
            return NULL;
        }

        // assign uniform locations
        GLuint uboIndex;
        GLuint program = config->GetProgram();
        g_transformBinding = 0;
        uboIndex = glGetUniformBlockIndex(program, "Transform");
        if (uboIndex != GL_INVALID_INDEX)
            glUniformBlockBinding(program, uboIndex, g_transformBinding);

        g_tessellationBinding = 1;
        uboIndex = glGetUniformBlockIndex(program, "Tessellation");
        if (uboIndex != GL_INVALID_INDEX)
            glUniformBlockBinding(program, uboIndex, g_tessellationBinding);

        g_lightingBinding = 2;
        uboIndex = glGetUniformBlockIndex(program, "Lighting");
        if (uboIndex != GL_INVALID_INDEX)
            glUniformBlockBinding(program, uboIndex, g_lightingBinding);

        // assign texture locations
        GLint loc;
        glUseProgram(program);

        if ((loc = glGetUniformLocation(program, "OsdPatchParamBuffer")) != -1) {
            glUniform1i(loc, 0); // GL_TEXTURE0
        }

        if (effectDesc.effect.paint) {
            if ((loc = glGetUniformLocation(program, "outTextureImage")) != -1) {
                glUniform1i(loc, 0); // image 0
            }
            if ((loc = glGetUniformLocation(program, "paintTexture")) != -1) {
                glUniform1i(loc, 5); // GL_TEXTURE5
            }
            if ((loc = glGetUniformLocation(program, "depthTexture")) != -1) {
                glUniform1i(loc, 6); // GL_TEXTURE6
            }
        } else {
            if ((loc = glGetUniformLocation(program, "textureImage_Data")) != -1) {
                glUniform1i(loc, 5); // GL_TEXTURE5
            }
            if ((loc = glGetUniformLocation(program, "textureImage_Packing")) != -1) {
                glUniform1i(loc, 6); // GL_TEXTURE6
            }
            if ((loc = glGetUniformLocation(program, "textureImage_Pages")) != -1) {
                glUniform1i(loc, 7); // GL_TEXTURE7
            }
        }

        glUseProgram(0);
        return config;
    }
};

ShaderCache g_shaderCache;

//------------------------------------------------------------------------------
static void
updateUniformBlocks() {
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
}

static void bindTextures(Effect effect) {
    if (effect.paint) {
        // set image
        glBindImageTexture(0, g_ptexTexels, 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32F);

        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_2D, g_paintTexture);

        glActiveTexture(GL_TEXTURE6);
        glBindTexture(GL_TEXTURE_2D, g_depthTexture);

        glActiveTexture(GL_TEXTURE0);
    } else {
        if (g_mesh->GetPatchTable()->GetPatchParamTextureBuffer()) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(
                GL_TEXTURE_BUFFER,
                g_mesh->GetPatchTable()->GetPatchParamTextureBuffer());
        }

        // color ptex
        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_2D_ARRAY, g_ptexTexels);

        glActiveTexture(GL_TEXTURE6);
        glBindTexture(GL_TEXTURE_BUFFER, g_ptexLayouts);

        glActiveTexture(GL_TEXTURE7);
        glBindTexture(GL_TEXTURE_BUFFER, g_ptexPages);
    }
    glActiveTexture(GL_TEXTURE0);
}

static GLuint
bindProgram(Effect effect, OpenSubdiv::Osd::PatchArray const & patch) {

    EffectDesc effectDesc(patch.GetDescriptor(), effect);

    // lookup shader cache (compile the shader if needed)
    GLDrawConfig *config = g_shaderCache.GetDrawConfig(effectDesc);
    if (!config) return 0;

    GLuint program = config->GetProgram();

    glUseProgram(program);

    GLint uniformImageSize = glGetUniformLocation(program, "imageSize");
    if (uniformImageSize >= 0)
        glUniform1i(uniformImageSize, g_pageSize);

    GLint uniformPrimitiveIdBase =
        glGetUniformLocation(program, "PrimitiveIdBase");
    if (uniformPrimitiveIdBase >= 0)
        glUniform1i(uniformPrimitiveIdBase, patch.GetPrimitiveIdBase());


    return program;
}

//------------------------------------------------------------------------------
static void
display() {

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, g_width, g_height);
    g_hud.FillBackground();

    // primitive counting
    glBeginQuery(GL_PRIMITIVES_GENERATED, g_primQuery);

    // prepare view matrix
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
    if (g_wire == 0) {
        glDisable(GL_CULL_FACE);
    }

    updateUniformBlocks();

    Effect effect;
    effect.color = g_displayColor;
    effect.displacement = g_displayDisplacement;
    effect.wire = g_wire;
    effect.paint = 0;

    bindTextures(effect);

    // make sure that the vertex buffer is interoped back as a GL resource.
    g_mesh->BindVertexBuffer();

    glBindVertexArray(g_vao);

    OpenSubdiv::Osd::PatchArrayVector const & patches =
        g_mesh->GetPatchTable()->GetPatchArrays();

    // patch drawing
    for (int i=0; i<(int)patches.size(); ++i) {
        OpenSubdiv::Osd::PatchArray const & patch = patches[i];
        OpenSubdiv::Far::PatchDescriptor desc = patch.GetDescriptor();

        GLenum primType = GL_PATCHES;
        glPatchParameteri(GL_PATCH_VERTICES, desc.GetNumControlVertices());

        GLuint program = bindProgram(effect, patch);
        GLuint diffuseColor = glGetUniformLocation(program, "diffuseColor");
        glProgramUniform4f(program, diffuseColor, 1, 1, 1, 1);

        glDrawElements(primType,
                       patch.GetNumPatches() * desc.GetNumControlVertices(),
                       GL_UNSIGNED_INT,
                       (void *)(patch.GetIndexBase() * sizeof(unsigned int)));
    }

    glBindVertexArray(0);
    glUseProgram(0);

    glEndQuery(GL_PRIMITIVES_GENERATED);

    glBindTexture(GL_TEXTURE_2D, g_depthTexture);
    glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, g_width, g_height);
    glBindTexture(GL_TEXTURE_2D, 0);

    if (g_wire == 0) {
        glEnable(GL_CULL_FACE);
    }

    GLuint numPrimsGenerated = 0;
    glGetQueryObjectuiv(g_primQuery, GL_QUERY_RESULT, &numPrimsGenerated);

    if (g_hud.IsVisible()) {
        g_fpsTimer.Stop();
        double fps = 1.0/g_fpsTimer.GetElapsed();
        g_fpsTimer.Start();
        // Average fps over a defined number of time samples for
        // easier reading in the HUD
        g_fpsTimeSamples[g_currentFpsTimeSample++] = float(fps);
        if (g_currentFpsTimeSample >= NUM_FPS_TIME_SAMPLES)
            g_currentFpsTimeSample = 0;
        double averageFps = 0;
        for (int i=0; i< NUM_FPS_TIME_SAMPLES; ++i) {
            averageFps += g_fpsTimeSamples[i]/(float)NUM_FPS_TIME_SAMPLES;
        }

        g_hud.DrawString(10, -100, "Tess level (+/-): %d", g_tessLevel);
        if (numPrimsGenerated > 1000000) {
            g_hud.DrawString(10, -80, "Primitives      : %3.1f million", (float)numPrimsGenerated/1000000.0);
        } else if (numPrimsGenerated > 1000) {
            g_hud.DrawString(10, -80, "Primitives      : %3.1f thousand", (float)numPrimsGenerated/1000.0);
        } else {
            g_hud.DrawString(10, -80, "Primitives      : %d", numPrimsGenerated);
        }
        g_hud.DrawString(10, -60, "Vertices        : %d", g_mesh->GetNumVertices());
        g_hud.DrawString(10, -20, "FPS             : %3.1f", averageFps);
    }

    g_hud.Flush();

    glFinish();

    //checkGLErrors("display leave");

    glfwSwapBuffers(g_window);
}

//------------------------------------------------------------------------------
void
drawStroke(int x, int y) {

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

    // make sure that the vertex buffer is interoped back as a GL resource.
    g_mesh->BindVertexBuffer();

    glBindVertexArray(g_vao);

    Effect effect;
    effect.color = 0;
    effect.displacement = g_displayDisplacement;
    effect.wire = 1;
    effect.paint = 1;
    bindTextures(effect);

    OpenSubdiv::Osd::PatchArrayVector const & patches =
        g_mesh->GetPatchTable()->GetPatchArrays();

    // patch drawing
    for (int i=0; i<(int)patches.size(); ++i) {

        OpenSubdiv::Osd::PatchArray const & patch = patches[i];
        OpenSubdiv::Far::PatchDescriptor desc = patch.GetDescriptor();

        GLenum primType = GL_PATCHES;
        glPatchParameteri(GL_PATCH_VERTICES, desc.GetNumControlVertices());

        bindProgram(effect, patch);

        glDrawElements(primType,
                       patch.GetNumPatches() * desc.GetNumControlVertices(),
                       GL_UNSIGNED_INT,
                       (void *)(patch.GetIndexBase() * sizeof(unsigned int)));
    }

    glBindVertexArray(0);
    glUseProgram(0);

    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                    GL_TEXTURE_FETCH_BARRIER_BIT);

    //checkGLErrors("display leave");
}

//------------------------------------------------------------------------------
static void
motion(GLFWwindow * w, double dx, double dy) {
    int x=(int)dx, y=(int)dy;

    if (glfwGetKey(w,GLFW_KEY_LEFT_ALT)) {

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
mouse(GLFWwindow * w, int button, int state, int /* mods */) {

    if (button == 0 && state == GLFW_PRESS && g_hud.MouseClick(g_prev_x, g_prev_y))
        return;

    if (button < 3) {
        g_mbutton[button] = (state == GLFW_PRESS);
    }

    if (! glfwGetKey(w, GLFW_KEY_LEFT_ALT)) {
        if (g_mbutton[0] && !g_mbutton[1] && !g_mbutton[2]) {
            drawStroke(g_prev_x, g_prev_y);
        }
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
    reshape(g_window, g_width, g_height);
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
keyboard(GLFWwindow *, int key, int /* scancode */, int event, int /* mods */) {

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
callbackWireframe(int b) {

    g_wire = b;
}

static void
callbackDisplay(bool /* checked */, int n) {

    if (n == 0) g_displayColor = !g_displayColor;
    else if (n == 1) g_displayDisplacement = !g_displayDisplacement;
}

static void
callbackLevel(int l) {

    g_level = l;
    createOsdMesh();
}

static void
callbackModel(int m) {

    if (m < 0)
        m = 0;

    if (m >= (int)g_defaultShapes.size())
        m = (int)g_defaultShapes.size() - 1;

    g_currentShape = m;
    createOsdMesh();
}

static void
callbackBrushSize(float value, int /* data */) {

    g_brushSize = (int)value;
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

    g_hud.AddCheckBox("Color (C)",  g_displayColor != 0, 10, 10, callbackDisplay, 0, 'c');
    g_hud.AddCheckBox("Displacement (D)",  g_displayDisplacement != 0, 10, 30, callbackDisplay, 1, 'd');

    int shading_pulldown = g_hud.AddPullDown("Shading (W)", 200, 10, 250, callbackWireframe, 'w');
    g_hud.AddPullDownButton(shading_pulldown, "Wire", 0, g_wire==0);
    g_hud.AddPullDownButton(shading_pulldown, "Shaded", 1, g_wire==1);
    g_hud.AddPullDownButton(shading_pulldown, "Wire+Shaded", 2, g_wire==2);

    g_hud.AddSlider("Brush size", 10.0f, 500.0f, (float)g_brushSize,
                     350, -60, 40, true, callbackBrushSize, 0);

    for (int i = 1; i < 11; ++i) {
        char level[16];
        sprintf(level, "Lv. %d", i);
        g_hud.AddRadioButton(3, level, i==g_level, 10, 170+i*20, callbackLevel, i, '0'+(i%10));
    }

    int pulldown_handle = g_hud.AddPullDown("Shape (N)", -300, 10, 300, callbackModel, 'n');
    for (int i = 0; i < (int)g_defaultShapes.size(); ++i) {
        g_hud.AddPullDownButton(pulldown_handle, g_defaultShapes[i].name.c_str(),i);
    }

    g_hud.Rebuild(g_width, g_height, frameBufferWidth, frameBufferHeight);
}

//------------------------------------------------------------------------------
static void
initGL() {

    glClearColor(0.1f, 0.1f, 0.1f, 0.0f);
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
}

//------------------------------------------------------------------------------
static void
idle() {
    updateGeom();
}

//------------------------------------------------------------------------------
static void
callbackError(OpenSubdiv::Far::ErrorType err, const char *message) {
    printf("OpenSubdiv Error: %d\n", err);
    printf("    %s\n", message);
}

//------------------------------------------------------------------------------
static void
callbackErrorGLFW(int error, const char* description) {
    fprintf(stderr, "GLFW Error (%d) : %s\n", error, description);
}

//------------------------------------------------------------------------------

int main(int argc, char ** argv) {

    ArgOptions args;

    args.Parse(argc, argv);
    args.PrintUnrecognizedArgsWarnings();

    g_yup = args.GetYUp();
    g_level = args.GetLevel();

    ViewerArgsUtils::PopulateShapes(args, &g_defaultShapes);

    initShapes();

    OpenSubdiv::Far::SetErrorCallback(callbackError);

    glfwSetErrorCallback(callbackErrorGLFW);
    if (! glfwInit()) {
        printf("Failed to initialize GLFW\n");
        return 1;
    }

    static const char windowTitle[] = "OpenSubdiv glPaintTest " OPENSUBDIV_VERSION_STRING;

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
        printf("Failed to open window.\n");
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(g_window);

    GLUtils::InitializeGL();
    GLUtils::PrintGLVersion();

    glfwSetKeyCallback(g_window, keyboard);
    glfwSetCursorPosCallback(g_window, motion);
    glfwSetMouseButtonCallback(g_window, mouse);
    glfwSetWindowCloseCallback(g_window, windowClose);

    initGL();

    // accommodate high DPI displays (e.g. mac retina displays)
    glfwGetFramebufferSize(g_window, &g_width, &g_height);
    glfwSetFramebufferSizeCallback(g_window, reshape);

    // as of GLFW 3.0.1 this callback is not implicit
    reshape();

    glfwSwapInterval(0);

    initHUD();
    callbackModel(g_currentShape);

    while (g_running) {
        idle();
        display();

        glfwPollEvents();
    }

    uninitGL();
    glfwTerminate();
}

//------------------------------------------------------------------------------
