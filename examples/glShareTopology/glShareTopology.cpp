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

#include "glLoader.h"

#include <GLFW/glfw3.h>
GLFWwindow* g_window=0;
GLFWmonitor* g_primary=0;

#include <opensubdiv/far/error.h>
#include <opensubdiv/far/stencilTable.h>
#include <opensubdiv/far/ptexIndices.h>

#include <opensubdiv/osd/mesh.h>
#include <opensubdiv/osd/glVertexBuffer.h>
#include <opensubdiv/osd/cpuGLVertexBuffer.h>
#include <opensubdiv/osd/cpuEvaluator.h>

#ifdef OPENSUBDIV_HAS_OPENMP
    #include <opensubdiv/osd/ompEvaluator.h>
#endif

#ifdef OPENSUBDIV_HAS_TBB
    #include <opensubdiv/osd/tbbEvaluator.h>
#endif

#ifdef OPENSUBDIV_HAS_OPENCL
    #include <opensubdiv/osd/clGLVertexBuffer.h>
    #include <opensubdiv/osd/clEvaluator.h>
    #include "../common/clDeviceContext.h"
    CLDeviceContext g_clDeviceContext;
#endif

#ifdef OPENSUBDIV_HAS_CUDA
    #include <opensubdiv/osd/cudaGLVertexBuffer.h>
    #include <opensubdiv/osd/cudaEvaluator.h>
    #include "../common/cudaDeviceContext.h"
    CudaDeviceContext g_cudaDeviceContext;
#endif

#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    #include <opensubdiv/osd/glXFBEvaluator.h>
#endif

#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    #include <opensubdiv/osd/glComputeEvaluator.h>
#endif


#include "../../regression/common/far_utils.h"
#include "init_shapes.h"

#include "../../regression/common/arg_utils.h"
#include "../common/stopwatch.h"
#include "../common/simple_math.h"
#include "../common/glHud.h"
#include "../common/glShaderCache.h"
#include "../common/glUtils.h"

#include <opensubdiv/osd/glslPatchShaderSource.h>
static const char *shaderSource =
#include "shader.gen.h"
;

#include <cfloat>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>

#include "scene.h"

SceneBase *g_scene = NULL;

using namespace OpenSubdiv;

// ---------------------------------------------------------------------------

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

enum HudCheckBox { kHUD_CB_ADAPTIVE,
                   kHUD_CB_MDI,
                   kHUD_CB_FREEZE,
                   kHUD_CB_VIEW_LOD,
                   kHUD_CB_PATCH_CULL };

// GUI variables
int   g_displayStyle = kShaded,
      g_MDI = 0,
      g_mbutton[3] = {0, 0, 0},
      g_freeze = 0,
      g_screenSpaceTess = 1,
      g_patchCull = 1,
      g_running = 1;

SceneBase::Options g_options;

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
int g_frame = 0;
int g_kernel = kCPU;
int g_numObjects = 64;
size_t g_vboSize = 0;
size_t g_iboSize = 0;

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

    int numObjects = g_scene->GetNumObjects();
    int column = (int)ceil(sqrt((float)numObjects));

    for (int i = 0; i < numObjects; ++i) {
        std::vector<float> const &restPosition = g_scene->GetRestPosition(i);

        int nverts = (int)restPosition.size()/3;
        int numVertexElements = (g_displayStyle == kVaryingInterleaved ? 7 : 3);
        int numVaryingElements = (g_displayStyle == kVarying ? 4 : 0);

        std::vector<float> vertex(numVertexElements * nverts);
        std::vector<float> varying(numVaryingElements * nverts);

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

        int vertsOffset = g_scene->GetVertsOffset(i);
        g_scene->UpdateVertexBuffer(vertsOffset, vertex);

        if (g_displayStyle == kVarying) {
            float *d = &varying[0];
            for (int j = 0; j < nverts; ++j) {
                *d++ = 1;
                *d++ = (1+(float)sin(0.1f*g_frame + i)) * 0.5f;
                *d++ = 1;
                *d++ = 1.0;
            }
            g_scene->UpdateVaryingBuffer(vertsOffset, varying);
        }
    }
}

static void
refine() {

    Stopwatch s;
    s.Start();

    int numObjects = g_scene->GetNumObjects();
    for (int i = 0; i < numObjects; ++i) {
        g_scene->Refine(i);
    }

    s.Stop();
    g_cpuTime = float(s.GetElapsed() * 1000.0f);
    s.Start();

    g_scene->Synchronize();

    s.Stop();
    g_gpuTime = float(s.GetElapsed() * 1000.0f);


    s.Stop();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
static void
fitFrame() {

    g_pan[0] = g_pan[1] = 0;
    g_dolly = g_size;
}

//------------------------------------------------------------------------------

union Effect {
    Effect(int displayStyle_,
           int screenSpaceTess_,
           int patchCull_) : value(0) {
        displayStyle = displayStyle_;
        screenSpaceTess = screenSpaceTess_;
        patchCull = patchCull_;
    }

    struct {
        unsigned int displayStyle: 3;
        unsigned int screenSpaceTess: 1;
        unsigned int patchCull: 1;
    };
    int value;

    bool operator < (const Effect &e) const {
        return value < e.value;
    }
};

static Effect
GetEffect() {

    return Effect(g_displayStyle, g_screenSpaceTess, g_patchCull);
}

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

class ShaderCache : public GLShaderCache<EffectDesc> {
public:
    virtual GLDrawConfig *CreateDrawConfig(EffectDesc const &effectDesc) {

        using namespace OpenSubdiv;

        // compile shader program
#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
        const char *glslVersion = "#version 400\n";
#else
        const char *glslVersion = "#version 330\n";
#endif
        GLDrawConfig *config = new GLDrawConfig(glslVersion);

        Far::PatchDescriptor::Type type = effectDesc.desc.GetType();

        std::string primTypeDefine =
            (type == Far::PatchDescriptor::QUADS ?
             "#define PRIM_QUAD\n" : "#define PRIM_TRI\n");

        // common defines
        std::stringstream ss;

        if (effectDesc.effect.screenSpaceTess) {
            ss << "#define OSD_ENABLE_SCREENSPACE_TESSELLATION\n";
        }
        if (effectDesc.effect.patchCull) {
            ss << "#define OSD_ENABLE_PATCH_CULL\n";
        }

        // display styles
        switch (effectDesc.effect.displayStyle) {
        case kWire:
            ss << "#define GEOMETRY_OUT_WIRE\n";
            break;
        case kWireShaded:
            ss << "#define GEOMETRY_OUT_LINE\n";
            break;
        case kShaded:
            ss << "#define GEOMETRY_OUT_FILL\n";
            break;
        case kVarying:
            ss << "#define VARYING_COLOR\n";
            ss << "#define GEOMETRY_OUT_FILL\n";
            break;
        case kVaryingInterleaved:
            ss << "#define VARYING_COLOR\n";
            ss << "#define GEOMETRY_OUT_FILL\n";
            break;
        }
        if (effectDesc.desc.IsAdaptive()) {
            ss << "#define SMOOTH_NORMALS\n";
        }

        // need for patch color-coding : we need these defines in the fragment shader
        if (type == Far::PatchDescriptor::GREGORY) {
            ss << "#define OSD_PATCH_GREGORY\n";
        } else if (type == Far::PatchDescriptor::GREGORY_BOUNDARY) {
            ss << "#define OSD_PATCH_GREGORY_BOUNDARY\n";
        } else if (type == Far::PatchDescriptor::GREGORY_BASIS) {
            ss << "#define OSD_PATCH_GREGORY_BASIS\n";
        } else if (type == Far::PatchDescriptor::LOOP) {
            ss << "#define OSD_PATCH_LOOP\n";
        } else if (type == Far::PatchDescriptor::GREGORY_TRIANGLE) {
            ss << "#define OSD_PATCH_GREGORY_TRIANGLE\n";
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
           << shaderSource
           << Osd::GLSLPatchShaderSource::GetVertexShaderSource(type);
        config->CompileAndAttachShader(GL_VERTEX_SHADER, ss.str());
        ss.str("");

        if (effectDesc.desc.IsAdaptive()) {
            // tess control shader
            ss << common
               << "#define OSD_PATCH_TESS_CONTROL_BSPLINE_SHADER\n"
               << shaderSource
               << Osd::GLSLPatchShaderSource::GetTessControlShaderSource(type);
            config->CompileAndAttachShader(GL_TESS_CONTROL_SHADER, ss.str());
            ss.str("");

            // tess eval shader
            ss << common
               << shaderSource
               << Osd::GLSLPatchShaderSource::GetTessEvalShaderSource(type);
            config->CompileAndAttachShader(GL_TESS_EVALUATION_SHADER, ss.str());
            ss.str("");
        }

        // geometry shader
        ss << common
           << "#define GEOMETRY_SHADER\n" // for my shader source
           << primTypeDefine
           << shaderSource;
        config->CompileAndAttachShader(GL_GEOMETRY_SHADER, ss.str());
        ss.str("");

        // fragment shader
        ss << common
           << "#define FRAGMENT_SHADER\n" // for my shader source
           << primTypeDefine
           << shaderSource;
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
        if ((loc = glGetUniformLocation(program, "OsdPatchParamBuffer")) != -1) {
            glProgramUniform1i(program, loc, 0); // GL_TEXTURE0
        }
        if ((loc = glGetUniformLocation(program, "OsdVertexBuffer")) != -1) {
            glProgramUniform1i(program, loc, 1); // GL_TEXTURE1
        }
        if ((loc = glGetUniformLocation(program, "OsdValenceBuffer")) != -1) {
            glProgramUniform1i(program, loc, 2); // GL_TEXTURE2
        }
        if ((loc = glGetUniformLocation(program, "OsdQuadOffsetBuffer")) != -1) {
            glProgramUniform1i(program, loc, 3); // GL_TEXTURE3
        }
        if ((loc = glGetUniformLocation(program, "OsdFVarDataBuffer")) != -1) {
            glProgramUniform1i(program, loc, 4); // GL_TEXTURE4
        }

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

static void
bindTextures() {

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, g_scene->GetPatchParamTexture());

    // XXX: LegacyGregory hasn't been supported.
    glActiveTexture(GL_TEXTURE0);
}

static GLenum
bindProgram(Effect effect,
            Far::PatchDescriptor desc,
            int basePrimitiveID) {

    EffectDesc effectDesc(desc, effect);

    typedef OpenSubdiv::Far::PatchDescriptor Descriptor;

    // lookup shader cache (compile the shader if needed)
    GLDrawConfig *config = g_shaderCache.GetDrawConfig(effectDesc);
    if (!config) return 0;

    GLuint program = config->GetProgram();

    glUseProgram(program);

    // bind standalone uniforms
    GLint uniformPrimitiveIdBase =
        glGetUniformLocation(program, "PrimitiveIdBase");
    if (uniformPrimitiveIdBase >=0)
        glUniform1i(uniformPrimitiveIdBase, basePrimitiveID);

    // return primtype
    GLenum primType;
    switch(effectDesc.desc.GetType()) {
    case Descriptor::QUADS:
        primType = GL_LINES_ADJACENCY;
        break;
    case Descriptor::TRIANGLES:
        primType = GL_TRIANGLES;
        break;
    default:
#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
        primType = GL_PATCHES;
        glPatchParameteri(GL_PATCH_VERTICES, effectDesc.desc.GetNumControlVertices());
#else
        primType = GL_POINTS;
#endif
        break;
    }

    return primType;
}


//------------------------------------------------------------------------------

template <typename T>
std::string formatWithCommas(T value) {
    std::stringstream ss;
    ss.imbue(std::locale(""));
    ss << std::fixed << value;
    return ss.str();
}

static void
display() {

    Stopwatch s;
    s.Start();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, g_width, g_height);
    g_hud.FillBackground();

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

    // make sure that the vertex buffer is interoped back as a GL resource.
    g_scene->BindVertexBuffer();

    glBindVertexArray(g_vao);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_scene->GetIndexBuffer());

    if (g_displayStyle == kVarying) {

        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, g_scene->BindVertexBuffer());
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 3, 0);

        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, g_scene->BindVaryingBuffer());
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 4, 0);

    } else if (g_displayStyle == kVaryingInterleaved) {

        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, g_scene->BindVertexBuffer());
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 7, 0);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 7,
                              (void*)(sizeof(GLfloat)*3));

    } else {

        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, g_scene->BindVertexBuffer());
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 3, 0);
        glDisableVertexAttribArray(1);
    }

    // update vertex buffer to texture for gregory patch drawing.
    //    g_topology->UpdateVertexTexture(g_vbo);

    int numDrawCalls = 0;
    // primitive counting
    glBeginQuery(GL_PRIMITIVES_GENERATED, g_queries[0]);
#if defined(GL_VERSION_3_3)
    glBeginQuery(GL_TIME_ELAPSED, g_queries[1]);
#endif

    updateUniformBlocks();
    bindTextures();

//#define ENABLE_MDI
#if defined(ENABLE_MDI) && defined(GL_ARB_multi_draw_indirect)
    if (g_MDI && glMultiDrawElementsIndirect) {
        SceneBase::BatchVector const &batches = g_scene->GetBatches();
        for (int i = 0; i < (int)batches.size(); ++i) {
            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, batches[i].dispatchBuffer);
            GLenum primType = bindProgram(GetEffect(),
                                          batches[i].desc,
                                          /*primitiveIDBase=*/0);
            glMultiDrawElementsIndirect(primType, GL_UNSIGNED_INT, 0,
                                        batches[i].count,
                                        batches[i].stride);
            // XXX: currently MDI path is broken because of the bad plumbing
            // of PrimitiveIdBase.
            ++numDrawCalls;
        }
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
    } else {
#else
    {
#endif
        int numObjects = g_scene->GetNumObjects();
        for (int i = 0; i < numObjects; ++i) {
            SceneBase::PatchArrayVector const &patchArrays = g_scene->GetPatchArrays(i);
            for (int j = 0; j < (int)patchArrays.size(); ++j) {
                SceneBase::PatchArray const &patchArray = patchArrays[j];

                int nPatch = patchArray.numPatches;
                int baseVertex = g_scene->GetVertsOffset(i);
                GLvoid *indices = (void *)(patchArray.indexOffset * sizeof(int));
                GLenum primType = bindProgram(GetEffect(),
                                              patchArray.desc,
                                              patchArray.primitiveIDOffset);
                glDrawElementsBaseVertex(
                    primType,
                    nPatch * patchArray.desc.GetNumControlVertices(),
                    GL_UNSIGNED_INT,
                    indices,
                    baseVertex);
                ++numDrawCalls;
            }
        }
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

    if (g_hud.IsVisible()) {
        g_fpsTimer.Stop();
        double fps = 1.0/g_fpsTimer.GetElapsed();
        g_fpsTimer.Start();

        g_hud.DrawString(230, -60, "Vertex + Varying Bufsize   : %s",
                         formatWithCommas(g_vboSize).c_str());
        g_hud.DrawString(230, -40, "Index + PatchParam Bufsize : %s",
                         formatWithCommas(g_iboSize).c_str());
        g_hud.DrawString(230, -20, "Stencil table size         : %s",
                         formatWithCommas(g_scene->GetStencilTableSize()).c_str());

        g_hud.DrawString(10, -180, "Tess level  : %d", g_tessLevel);
        g_hud.DrawString(10, -160, "Primitives  : %s", formatWithCommas(numPrimsGenerated).c_str());
        g_hud.DrawString(10, -140, "Draw calls  : %d", numDrawCalls);
        g_hud.DrawString(10, -100, "GPU Compute : %.3f ms", g_gpuTime);
        g_hud.DrawString(10, -80,  "CPU Compute : %.3f ms", g_cpuTime);
        g_hud.DrawString(10, -60,  "GPU Draw    : %.3f ms", drawGpuTime);
        g_hud.DrawString(10, -40,  "CPU Draw    : %.3f ms", drawCpuTime);
        g_hud.DrawString(10, -20,  "FPS         : %3.1f", fps);

        g_hud.Flush();
    }

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
uninitGL() {

    glDeleteQueries(2, g_queries);
    glDeleteVertexArrays(1, &g_vao);

    delete g_scene;
    g_scene = NULL;
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
rebuildObjects() {

    // create Objects

    int numVerts = g_scene->AddObjects(g_numObjects);

    Osd::BufferDescriptor vertexDesc, varyingDesc;
    bool interleaved = true;

    if (g_displayStyle == kVaryingInterleaved) {
        vertexDesc = Osd::BufferDescriptor(0, 3, 7);
        varyingDesc = Osd::BufferDescriptor(3, 4, 7);
        interleaved = true;
    } else if (g_displayStyle == kVarying) {
        vertexDesc = Osd::BufferDescriptor(0, 3, 3);
        varyingDesc = Osd::BufferDescriptor(0, 4, 4);
        interleaved = false;
    } else {
        vertexDesc = Osd::BufferDescriptor(0, 3, 3);
        varyingDesc = Osd::BufferDescriptor(0, 0, 0);
        interleaved = false;
    }

    g_vboSize = g_scene->AllocateVBO(numVerts, vertexDesc, varyingDesc, interleaved);

    updateGeom();
    refine();
}

static void
rebuildTopology() {

    if (g_scene) delete g_scene;

    if (g_kernel == kCPU) {
        g_scene = new Scene<Osd::CpuEvaluator,
                            Osd::CpuGLVertexBuffer,
                            Far::StencilTable>(g_options);
#ifdef OPENSUBDIV_HAS_OPENMP
    } else if (g_kernel == kOPENMP) {
        g_scene = new Scene<Osd::OmpEvaluator,
                            Osd::CpuGLVertexBuffer,
                            Far::StencilTable>(g_options);
#endif
#ifdef OPENSUBDIV_HAS_TBB
    } else if (g_kernel == kTBB) {
        g_scene = new Scene<Osd::TbbEvaluator,
                            Osd::CpuGLVertexBuffer,
                            Far::StencilTable>(g_options);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    } else if (g_kernel == kCUDA) {
        g_scene = new Scene<Osd::CudaEvaluator,
                            Osd::CudaGLVertexBuffer,
                            Osd::CudaStencilTable>(g_options);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    } else if (g_kernel == kCL) {
        static Osd::EvaluatorCacheT<Osd::CLEvaluator> clEvaluatorCache;
        g_scene = new Scene<Osd::CLEvaluator,
                            Osd::CLGLVertexBuffer,
                            Osd::CLStencilTable,
                            CLDeviceContext>(g_options, &clEvaluatorCache,
                                             &g_clDeviceContext);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    } else if (g_kernel == kGLSL) {
        static Osd::EvaluatorCacheT<Osd::GLXFBEvaluator> glXFBEvaluatorCache;
        g_scene = new Scene<Osd::GLXFBEvaluator,
                            Osd::GLVertexBuffer,
                            Osd::GLStencilTableTBO>(g_options, &glXFBEvaluatorCache);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    } else if (g_kernel == kGLSLCompute) {
        static Osd::EvaluatorCacheT<Osd::GLComputeEvaluator> glComputeEvaluatorCache;
        g_scene = new Scene<Osd::GLComputeEvaluator,
                            Osd::GLVertexBuffer,
                            Osd::GLStencilTableSSBO>(g_options, &glComputeEvaluatorCache);
#endif
    }

    for (int i = 0; i < (int)g_defaultShapes.size(); ++i) {
        Shape const * shape = Shape::parseObj(g_defaultShapes[i]);

        bool varying = (g_displayStyle==kVarying || g_displayStyle==kVaryingInterleaved);
        g_scene->AddTopology(shape, g_level, varying);

        delete shape;
    }

    g_iboSize = g_scene->CreateIndexBuffer();

    rebuildObjects();
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
        case '.': g_numObjects *= 2; rebuildObjects(); break;
        case ',': g_numObjects = std::max(1, g_numObjects/2); rebuildObjects(); break;
        case GLFW_KEY_ESCAPE: g_hud.SetVisible(!g_hud.IsVisible()); break;
    }
}

//------------------------------------------------------------------------------

static void
callbackEndCap(int endCap) {
    g_options.endCap = endCap;
    rebuildTopology();
}

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

    rebuildTopology();
}

static void
callbackLevel(int l) {

    g_level = l;
    rebuildTopology();
}

static void
callbackSlider(float value, int /* data */) {

    g_numObjects = (int)value;
    rebuildObjects();
}

static void
callbackDisplayStyle(int b) {

    g_displayStyle = b;
    rebuildTopology();
}

static void
callbackCheckBox(bool checked, int button) {

    switch (button) {
    case kHUD_CB_ADAPTIVE:
        g_options.adaptive = checked;
        rebuildTopology();
        break;
    case kHUD_CB_MDI:
        g_MDI = checked;
        break;
    case kHUD_CB_VIEW_LOD:
        g_screenSpaceTess = checked;
        break;
    case kHUD_CB_PATCH_CULL:
        g_patchCull = checked;
        break;
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

    int shading_pulldown = g_hud.AddPullDown("Shading (W)", 10, 10, 250, callbackDisplayStyle, 'w');
    g_hud.AddPullDownButton(shading_pulldown, "Wire", kWire, g_displayStyle==kWire);
    g_hud.AddPullDownButton(shading_pulldown, "Shaded", kShaded, g_displayStyle==kShaded);
    g_hud.AddPullDownButton(shading_pulldown, "Wire+Shaded", kWireShaded, g_displayStyle==kWireShaded);
    g_hud.AddPullDownButton(shading_pulldown, "Varying", kVarying, g_displayStyle==kVarying);
    g_hud.AddPullDownButton(shading_pulldown, "Varying(Interleaved)", kVaryingInterleaved, g_displayStyle==kVaryingInterleaved);

    g_hud.AddCheckBox("Screen space LOD (V)",  g_screenSpaceTess != 0,
                      10, 110, callbackCheckBox, kHUD_CB_VIEW_LOD, 'v');
    g_hud.AddCheckBox("Frustum Patch Culling (B)",  g_patchCull != 0,
                      10, 130, callbackCheckBox, kHUD_CB_PATCH_CULL, 'b');
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
    if (CLDeviceContext::HAS_CL_VERSION_1_1()) {
        g_hud.AddPullDownButton(compute_pulldown, "OpenCL", kCL);
    }
#endif
#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    g_hud.AddPullDownButton(compute_pulldown, "GLSL TransformFeedback", kGLSL);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    if (GLUtils::GL_ARBComputeShaderOrGL_VERSION_4_3()) {
        g_hud.AddPullDownButton(compute_pulldown, "GLSL Compute", kGLSLCompute);
    }
#endif

    g_hud.AddSlider("Objects count", 1, 1000, 25,
                    -200, 20, 20, true, callbackSlider, 0);

    {
#if defined(ENABLE_MDI) && defined(GL_ARB_multi_draw_indirect)
        g_hud.AddCheckBox("Multi Draw Indirect (m)", g_MDI != 0,
                          10, 170, callbackCheckBox, kHUD_CB_MDI, 'm');
#endif
        g_hud.AddCheckBox("Adaptive (`)", g_options.adaptive != 0,
                          10, 190, callbackCheckBox, kHUD_CB_ADAPTIVE, '`');

        int endcap_pulldown = g_hud.AddPullDown(
            "End cap (E)", 10, 210, 200, callbackEndCap, 'e');
        g_hud.AddPullDownButton(endcap_pulldown, "Regular",
                                SceneBase::kEndCapBSplineBasis,
                                g_options.endCap == SceneBase::kEndCapBSplineBasis);
        g_hud.AddPullDownButton(endcap_pulldown, "Gregory",
                                SceneBase::kEndCapGregoryBasis,
                                g_options.endCap == SceneBase::kEndCapGregoryBasis);
    }


    for (int i = 1; i < 11; ++i) {
        char level[16];
        sprintf(level, "Lv. %d", i);
        g_hud.AddRadioButton(3, level, i==g_level, 10, 210+i*20, callbackLevel, i, '0'+(i%10));
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
    if (! g_freeze) {
        ++g_frame;
        updateGeom();
        refine();
    }
}

//------------------------------------------------------------------------------
static void
callbackError(Far::ErrorType err, const char *message) {
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

    g_options.adaptive = args.GetAdaptive();
    g_level = args.GetLevel();

    Far::SetErrorCallback(callbackError);

    glfwSetErrorCallback(callbackErrorGLFW);
    if (! glfwInit()) {
        printf("Failed to initialize GLFW\n");
        return 1;
    }

    static const char windowTitle[] = "OpenSubdiv batching example " OPENSUBDIV_VERSION_STRING;

    GLUtils::SetMinimumGLVersion();

    if (! (g_window=glfwCreateWindow(g_width, g_height, windowTitle, NULL, NULL))) {
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

    initShapes();
    initGL();

    glfwSwapInterval(0);

    initHUD();
    rebuildTopology();

    while (g_running) {
        idle();
        display();

        glfwPollEvents();
        glfwSwapBuffers(g_window);
    }

    uninitGL();
    glfwTerminate();
}

//------------------------------------------------------------------------------
