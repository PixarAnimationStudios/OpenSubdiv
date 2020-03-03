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
GLFWwindow* g_window = 0;
GLFWmonitor* g_primary = 0;

#include <opensubdiv/far/error.h>
#include <opensubdiv/osd/cpuEvaluator.h>
#include <opensubdiv/osd/cpuVertexBuffer.h>
#include <opensubdiv/osd/cpuGLVertexBuffer.h>
#include <opensubdiv/osd/glMesh.h>
OpenSubdiv::Osd::GLMeshInterface *g_mesh = NULL;

#include "../../regression/common/far_utils.h"
#include "../../regression/common/arg_utils.h"
#include "../common/stopwatch.h"
#include "../common/simple_math.h"
#include "../common/glControlMeshDisplay.h"
#include "../common/glHud.h"
#include "../common/glShaderCache.h"
#include "../common/glUtils.h"
#include "../common/viewerArgsUtils.h"

#include <opensubdiv/osd/glslPatchShaderSource.h>
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

enum DisplayStyle { kWire = 0,
                    kShaded,
                    kWireShaded };

enum EndCap       { kEndCapBilinearBasis,
                    kEndCapBSplineBasis,
                    kEndCapGregoryBasis };

int g_currentShape = 0;

int   g_frame = 0,
      g_repeatCount = 0;

OpenSubdiv::Sdc::Options::FVarLinearInterpolation  g_fvarInterp =
    OpenSubdiv::Sdc::Options::FVAR_LINEAR_ALL;

// GUI variables
int   g_freeze = 0,
      g_uvCullBackface = 0,
      g_displayStyle = kWireShaded,
      g_adaptive = 1,
      g_smoothCornerPatch = 1,
      g_singleCreasePatch = 1,
      g_infSharpPatch = 1,
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

bool  g_yup = false;

int   g_prev_x = 0,
      g_prev_y = 0;

int   g_width = 1600,
      g_height = 800;

GLhud g_hud;
GLControlMeshDisplay g_controlMeshDisplay;

// geometry
std::vector<float> g_orgPositions,
                   g_positions,
                   g_normals;

Scheme             g_scheme;

int g_endCap = kEndCapGregoryBasis;
int g_level = 2;
int g_tessLevel = 1;
int g_tessLevelMin = 1;

GLuint g_transformUB = 0,
       g_transformBinding = 0,
       g_tessellationUB = 0,
       g_tessellationBinding = 0,
       g_fvarArrayDataUB = 0,
       g_fvarArrayDataBinding = 0;

struct Transform {
    float ModelViewMatrix[16];
    float ProjectionMatrix[16];
    float ModelViewProjectionMatrix[16];
    float ModelViewInverseMatrix[16];
    float UvViewMatrix[16];
} g_transformData;

GLuint g_vao = 0;

std::vector<int> g_coarseEdges;
std::vector<float> g_coarseEdgeSharpness;
std::vector<float> g_coarseVertexSharpness;

struct Program {
    GLuint program;
    GLuint uniformModelViewProjectionMatrix;
    GLuint attrPosition;
    GLuint attrColor;
} g_defaultProgram;

struct FVarData
{
    FVarData() :
        textureBuffer(0), textureParamBuffer(0) {
    }
    ~FVarData() {
        Release();
    }
    void Release() {
        if (textureBuffer)
            glDeleteTextures(1, &textureBuffer);
        textureBuffer = 0;
        if (textureParamBuffer)
            glDeleteTextures(1, &textureParamBuffer);
        textureParamBuffer = 0;
    }
    void Create(OpenSubdiv::Far::TopologyRefiner const *refiner,
                OpenSubdiv::Far::PatchTable const *patchTable,
                std::vector<float> const & fvarSrcData,
                int fvarWidth, int fvarChannel = 0) {

        using namespace OpenSubdiv;

        Release();

        Far::StencilTableFactory::Options soptions;
        soptions.interpolationMode = Far::StencilTableFactory::INTERPOLATE_FACE_VARYING;
        soptions.fvarChannel = fvarChannel;
        soptions.generateOffsets = true;
        soptions.generateIntermediateLevels = !refiner->IsUniform();
        Far::StencilTable const *fvarStencils =
            Far::StencilTableFactory::Create(*refiner, soptions);

        if (Far::StencilTable const *fvarStencilsWithLocalPoints =
            Far::StencilTableFactory::AppendLocalPointStencilTableFaceVarying(
                *refiner,
                fvarStencils,
                patchTable->GetLocalPointFaceVaryingStencilTable(),
                fvarChannel)) {
            delete fvarStencils;
            fvarStencils = fvarStencilsWithLocalPoints;
        }

        int numSrcFVarPoints = (int)fvarSrcData.size() / fvarWidth;
        int numFVarPoints = numSrcFVarPoints
                          + fvarStencils->GetNumStencils();

        Osd::CpuVertexBuffer *fvarBuffer =
            Osd::CpuVertexBuffer::Create(fvarWidth, numFVarPoints);
        fvarBuffer->UpdateData(&fvarSrcData[0], 0, numSrcFVarPoints);

        Osd::BufferDescriptor srcDesc(0, fvarWidth, fvarWidth);
        Osd::BufferDescriptor dstDesc(numSrcFVarPoints*fvarWidth,
                                      fvarWidth, fvarWidth);

        Osd::CpuEvaluator::EvalStencils(fvarBuffer, srcDesc,
                                        fvarBuffer, dstDesc,
                                        fvarStencils);

        Far::ConstIndexArray indices = patchTable->GetFVarValues();
        const float * fvarSrcDataPtr = !refiner->IsUniform()
            ? fvarBuffer->BindCpuBuffer()
            : fvarBuffer->BindCpuBuffer() + numSrcFVarPoints * fvarWidth;

        // expand fvardata to per-patch array
        std::vector<float> data;
        data.reserve(indices.size() * fvarWidth);

        for (int fvert = 0; fvert < (int)indices.size(); ++fvert) {
            int index = indices[fvert] * fvarWidth;
            for (int i = 0; i < fvarWidth; ++i) {
                data.push_back(fvarSrcDataPtr[index++]);
            }
        }
        GLuint buffer;
        glGenBuffers(1, &buffer);
        glBindBuffer(GL_ARRAY_BUFFER, buffer);
        glBufferData(GL_ARRAY_BUFFER, data.size()*sizeof(float),
                     &data[0], GL_STATIC_DRAW);

        delete fvarBuffer;

        glGenTextures(1, &textureBuffer);
        glBindTexture(GL_TEXTURE_BUFFER, textureBuffer);
        glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, buffer);
        glBindTexture(GL_TEXTURE_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glDeleteBuffers(1, &buffer);

        Far::ConstPatchParamArray fvarParam = patchTable->GetFVarPatchParams();

        glGenBuffers(1, &buffer);
        glBindBuffer(GL_ARRAY_BUFFER, buffer);
        glBufferData(GL_ARRAY_BUFFER, fvarParam.size()*sizeof(Far::PatchParam),
                     &fvarParam[0], GL_STATIC_DRAW);

        glGenTextures(1, &textureParamBuffer);
        glBindTexture(GL_TEXTURE_BUFFER, textureParamBuffer);
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RG32I, buffer);
        glBindTexture(GL_TEXTURE_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glDeleteBuffers(1, &buffer);
    }
    GLuint textureBuffer, textureParamBuffer;
} g_fvarData;

//------------------------------------------------------------------------------
static GLuint
compileShader(GLenum shaderType, const char *source) {

    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    return shader;
}

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

#include "init_shapes.h"


//------------------------------------------------------------------------------
static void
calcNormals(OpenSubdiv::Far::TopologyRefiner const & refiner,
    std::vector<float> const & pos, std::vector<float> & normals) {

    typedef OpenSubdiv::Far::ConstIndexArray IndexArray;

    OpenSubdiv::Far::TopologyLevel const & refBaseLevel = refiner.GetLevel(0);

    // calc normal vectors
    int nverts = (int)pos.size()/3;

    int nfaces = refBaseLevel.GetNumFaces();
    for (int face = 0; face < nfaces; ++face) {

        IndexArray fverts = refBaseLevel.GetFaceVertices(face);

        assert(fverts.size()>=2);

        float const * p0 = &pos[fverts[0]*3],
                    * p1 = &pos[fverts[1]*3],
                    * p2 = &pos[fverts[2]*3];

        float n[3];
        cross(n, p0, p1, p2);

        for (int j = 0; j < fverts.size(); j++) {
            int idx = fverts[j] * 3;
            normals[idx  ] += n[0];
            normals[idx+1] += n[1];
            normals[idx+2] += n[2];
        }
    }

    for (int i = 0; i < nverts; ++i)
        normalize(&normals[i*3]);
}

//------------------------------------------------------------------------------
static void
updateGeom() {

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
rebuildMesh() {
    ShapeDesc const &shapeDesc = g_defaultShapes[g_currentShape];
    int level = g_level;
    Scheme scheme = g_defaultShapes[g_currentShape].scheme;

    Shape * shape = Shape::parseObj(shapeDesc);

    if (!shape->HasUV()) {
        printf("Error: shape %s does not contain face-varying UVs\n", shapeDesc.name.c_str());
        exit(1);
    }

    // create Far mesh (topology)
    OpenSubdiv::Sdc::SchemeType sdctype = GetSdcType(*shape);
    OpenSubdiv::Sdc::Options sdcoptions = GetSdcOptions(*shape);

    sdcoptions.SetFVarLinearInterpolation(g_fvarInterp);

    OpenSubdiv::Far::TopologyRefiner * refiner =
        OpenSubdiv::Far::TopologyRefinerFactory<Shape>::Create(*shape,
            OpenSubdiv::Far::TopologyRefinerFactory<Shape>::Options(sdctype, sdcoptions));

    // save coarse topology (used for coarse mesh drawing)
    g_controlMeshDisplay.SetTopology(refiner->GetLevel(0));

    g_orgPositions=shape->verts;
    g_normals.resize(g_orgPositions.size(), 0.0f);
    calcNormals(*refiner, g_orgPositions, g_normals);

    g_positions.resize(g_orgPositions.size(),0.0f);

    g_scheme = scheme;

    OpenSubdiv::Osd::MeshBitset bits;
    bits.set(OpenSubdiv::Osd::MeshAdaptive, g_adaptive != 0);
    bits.set(OpenSubdiv::Osd::MeshUseSmoothCornerPatch, g_smoothCornerPatch != 0);
    bits.set(OpenSubdiv::Osd::MeshUseSingleCreasePatch, g_singleCreasePatch != 0);
    bits.set(OpenSubdiv::Osd::MeshUseInfSharpPatch, g_infSharpPatch != 0);
    bits.set(OpenSubdiv::Osd::MeshFVarData, 1);
    bits.set(OpenSubdiv::Osd::MeshFVarAdaptive, 1);
    bits.set(OpenSubdiv::Osd::MeshEndCapBilinearBasis, g_endCap == kEndCapBilinearBasis);
    bits.set(OpenSubdiv::Osd::MeshEndCapBSplineBasis, g_endCap == kEndCapBSplineBasis);
    bits.set(OpenSubdiv::Osd::MeshEndCapGregoryBasis, g_endCap == kEndCapGregoryBasis);


    int numVertexElements = 3;
    int numVaryingElements = 0;

    delete g_mesh;
    g_mesh = new OpenSubdiv::Osd::Mesh<OpenSubdiv::Osd::CpuGLVertexBuffer,
                                       OpenSubdiv::Far::StencilTable,
                                       OpenSubdiv::Osd::CpuEvaluator,
                                       OpenSubdiv::Osd::GLPatchTable>(
                                           refiner,
                                           numVertexElements,
                                           numVaryingElements,
                                           level, bits);

    // set fvardata to texture buffer
    g_fvarData.Create(refiner, g_mesh->GetFarPatchTable(),
                      shape->uvs, shape->GetFVarWidth());

    delete shape;

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

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_mesh->GetPatchTable()->GetPatchIndexBuffer());
    glBindBuffer(GL_ARRAY_BUFFER, g_mesh->BindVertexBuffer());

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 3, 0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

//------------------------------------------------------------------------------
static void
fitFrame() {

    g_pan[0] = g_pan[1] = 0;
    g_dolly = g_size;
    g_uvPan[0] = g_uvPan[1] = 0;
    g_uvScale = 1.0;
}

//------------------------------------------------------------------------------

union Effect {

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

static Effect
GetEffect(bool uvDraw = false) {

    return Effect(g_displayStyle, uvDraw);
}

// ---------------------------------------------------------------------------

struct EffectDesc {
    EffectDesc(OpenSubdiv::Far::PatchDescriptor desc,
               Effect effect) : desc(desc),
                                effect(effect),
                                maxValence(0), numElements(0) { }

    OpenSubdiv::Far::PatchDescriptor desc;
    Effect effect;
    int maxValence;
    int numElements;

    bool operator < (const EffectDesc &e) const {
        return
            (desc < e.desc || ((desc == e.desc &&
            (maxValence < e.maxValence || ((maxValence == e.maxValence) &&
            (numElements < e.numElements || ((numElements == e.numElements) &&
            (effect < e.effect))))))));
    }
};

// ---------------------------------------------------------------------------

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

        // common defines
        std::stringstream ss;

        if (type == Far::PatchDescriptor::QUADS) {
            ss << "#define PRIM_QUAD\n";
        } else {
            ss << "#define PRIM_TRI\n";
        }

        if (effectDesc.effect.uvDraw) {
            ss << "#define GEOMETRY_OUT_FILL\n";
            ss << "#define GEOMETRY_UV_VIEW\n";
        } else {
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
            }
        }

        // for legacy gregory
        ss << "#define OSD_MAX_VALENCE " << effectDesc.maxValence << "\n";
        ss << "#define OSD_NUM_ELEMENTS " << effectDesc.numElements << "\n";

        // face varying width
        ss << "#define OSD_FVAR_WIDTH 2\n";

        if (! effectDesc.desc.IsAdaptive()) {
            ss << "#define SHADING_FACEVARYING_UNIFORM_SUBDIVISION\n";
        }

        // include osd PatchCommon
        ss << "#define OSD_PATCH_BASIS_GLSL\n";
        ss << Osd::GLSLPatchShaderSource::GetPatchBasisShaderSource();
        ss << Osd::GLSLPatchShaderSource::GetCommonShaderSource();
        std::string common = ss.str();
        ss.str("");

        // vertex shader
        ss << common
            // enable local vertex shader
           << (effectDesc.desc.IsAdaptive() ? "" : "#define VERTEX_SHADER\n")
           << shaderSource
           << Osd::GLSLPatchShaderSource::GetVertexShaderSource(type);
        config->CompileAndAttachShader(GL_VERTEX_SHADER, ss.str());
        ss.str("");

        if (effectDesc.desc.IsAdaptive()) {
            // tess control shader
            ss << common
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
           << "#define GEOMETRY_SHADER\n"
           << shaderSource;
        config->CompileAndAttachShader(GL_GEOMETRY_SHADER, ss.str());
        ss.str("");

        // fragment shader
        ss << common
           << "#define FRAGMENT_SHADER\n"
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

        g_fvarArrayDataBinding = 2;
        uboIndex = glGetUniformBlockIndex(program, "OsdFVarArrayData");
        if (uboIndex != GL_INVALID_INDEX)
            glUniformBlockBinding(program, uboIndex, g_fvarArrayDataBinding);

        // assign texture locations
        GLint loc;
        glUseProgram(program);
        if ((loc = glGetUniformLocation(program, "OsdPatchParamBuffer")) != -1) {
            glUniform1i(loc, 0); // GL_TEXTURE0
        }
        if ((loc = glGetUniformLocation(program, "OsdFVarDataBuffer")) != -1) {
            glUniform1i(loc, 1); // GL_TEXTURE1
        }
        if ((loc = glGetUniformLocation(program, "OsdFVarParamBuffer")) != -1) {
            glUniform1i(loc, 2); // GL_TEXTURE2
        }


        return config;
    }
};

ShaderCache g_shaderCache;

//------------------------------------------------------------------------------
static void
updateUniformBlocks() {

    using namespace OpenSubdiv;

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

    // Update and bind fvar patch array state
    Osd::PatchArrayVector const &fvarPatchArrays =
        g_mesh->GetPatchTable()->GetFVarPatchArrays();
    if (! fvarPatchArrays.empty()) {
	// bind patch arrays UBO (std140 struct size padded to vec4 alignment)
	int patchArraySize =
	    sizeof(GLint) * ((sizeof(Osd::PatchArray)/sizeof(GLint) + 3) & ~3);
        if (!g_fvarArrayDataUB) {
            glGenBuffers(1, &g_fvarArrayDataUB);
        }
	glBindBuffer(GL_UNIFORM_BUFFER, g_fvarArrayDataUB);
	glBufferData(GL_UNIFORM_BUFFER,
	    fvarPatchArrays.size()*patchArraySize, NULL, GL_STATIC_DRAW);
	for (int i=0; i<(int)fvarPatchArrays.size(); ++i) {
	    glBufferSubData(GL_UNIFORM_BUFFER,
		i*patchArraySize, sizeof(Osd::PatchArray), &fvarPatchArrays[i]);
	}

        glBindBufferBase(GL_UNIFORM_BUFFER,
                g_fvarArrayDataBinding, g_fvarArrayDataUB);
    }
}

static void
bindTextures() {
    if (g_mesh->GetPatchTable()->GetPatchParamTextureBuffer()) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_BUFFER,
            g_mesh->GetPatchTable()->GetPatchParamTextureBuffer());
    }
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_BUFFER, g_fvarData.textureBuffer);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_BUFFER, g_fvarData.textureParamBuffer);

    glActiveTexture(GL_TEXTURE0);
}

static GLenum
bindProgram(Effect effect,
            OpenSubdiv::Osd::PatchArray const & patch) {

    EffectDesc effectDesc(patch.GetDescriptor(), effect);

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
        glUniform1i(uniformPrimitiveIdBase, patch.GetPrimitiveIdBase());

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
static void
display() {

    Stopwatch s;
    s.Start();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, g_width/2, g_height);
    g_hud.FillBackground();

    // prepare view matrix
    double aspect = (g_width/2)/(double)g_height;
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

    identity(g_transformData.UvViewMatrix);
    scale(g_transformData.UvViewMatrix, g_uvScale, g_uvScale, 1);
    translate(g_transformData.UvViewMatrix, -g_uvPan[0], -g_uvPan[1], 0);

    glEnable(GL_DEPTH_TEST);

    // make sure that the vertex buffer is interoped back as a GL resource.
    GLuint vbo = g_mesh->BindVertexBuffer();

    glBindVertexArray(g_vao);

    OpenSubdiv::Osd::PatchArrayVector const & patches =
        g_mesh->GetPatchTable()->GetPatchArrays();

    if (g_displayStyle != kWire)
        glEnable(GL_CULL_FACE);

    updateUniformBlocks();
    bindTextures();

    // patch drawing
    for (int i = 0; i < (int)patches.size(); ++i) {
        OpenSubdiv::Osd::PatchArray const & patch = patches[i];

        GLenum primType = bindProgram(GetEffect(), patch);

        glDrawElements(
            primType,
            patch.GetNumPatches()*patch.GetDescriptor().GetNumControlVertices(),
            GL_UNSIGNED_INT,
            (void *)(patch.GetIndexBase() * sizeof(unsigned int)));
    }
    if (g_displayStyle != kWire)
        glDisable(GL_CULL_FACE);

    glBindVertexArray(0);
    glUseProgram(0);

    // draw the control mesh
    g_controlMeshDisplay.Draw(vbo, 3*sizeof(float),
                              g_transformData.ModelViewProjectionMatrix);

    // ---------------------------------------------
    // uv viewport
    glViewport(g_width/2, 0, g_width/2, g_height);

    g_mesh->BindVertexBuffer();

    glBindVertexArray(g_vao);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    if (g_uvCullBackface)
        glEnable(GL_CULL_FACE);

    for (int i = 0; i < (int)patches.size(); ++i) {
        OpenSubdiv::Osd::PatchArray const & patch = patches[i];

        GLenum primType = bindProgram(GetEffect(/*uvDraw=*/ true), patch);

        glDrawElements(
            primType,
            patch.GetNumPatches()*patch.GetDescriptor().GetNumControlVertices(),
            GL_UNSIGNED_INT,
            (void *)(patch.GetIndexBase() * sizeof(unsigned int)));
    }

    if (g_uvCullBackface)
        glDisable(GL_CULL_FACE);

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
motion(GLFWwindow *, double dx, double dy) {
    int x=(int)dx, y=(int)dy;

    if (g_mouseUvView) {
        if (!g_mbutton[0] && !g_mbutton[1] && g_mbutton[2]) {
            // pan
            g_uvPan[0] -= (x - g_prev_x) * 2 / g_uvScale / static_cast<float>(g_width/2);
            g_uvPan[1] += (y - g_prev_y) * 2 / g_uvScale / static_cast<float>(g_height);
        } else if ((g_mbutton[0] && !g_mbutton[1] && g_mbutton[2]) ||
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
        } else if ((g_mbutton[0] && !g_mbutton[1] && g_mbutton[2]) ||
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
mouse(GLFWwindow *, int button, int state, int /* mods */) {

    if (button == 0 && state == GLFW_PRESS && g_hud.MouseClick(g_prev_x, g_prev_y))
        return;

    if (button < 3) {
        g_mbutton[button] = (state == GLFW_PRESS);
    }

    // window size might not match framebuffer size on a high DPI display
    int windowWidth = g_width, windowHeight = g_height;
    glfwGetWindowSize(g_window, &windowWidth, &windowHeight);

    g_mouseUvView = (g_prev_x > windowWidth/2);
}

//------------------------------------------------------------------------------
static void
uninitGL() {

    glDeleteVertexArrays(1, &g_vao);
    if (g_mesh)
        delete g_mesh;
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
keyboard(GLFWwindow *, int key, int /* scancode */, int event, int /* mods */) {

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
callbackDisplayStyle(int b) {

    g_displayStyle = b;
}

static void
callbackEndCap(int endCap) {
    g_endCap = endCap;
    rebuildMesh();
}

static void
callbackLevel(int l) {

    g_level = l;
    rebuildMesh();
}

static void
callbackModel(int m) {

    int maxShapes = static_cast<int>(g_defaultShapes.size());
    g_currentShape = std::max(0, std::min(m, maxShapes-1));
    rebuildMesh();
}

static void
callbackControlEdges(bool checked, int /* a */) {

    g_controlMeshDisplay.SetEdgesDisplay(checked);
    rebuildMesh();
}

static void
callbackControlVertices(bool checked, int /* a */) {

    g_controlMeshDisplay.SetVerticesDisplay(checked);
    rebuildMesh();
}

static void
callbackUVCullBackface(bool checked, int /* a */) {

    g_uvCullBackface = checked;
    rebuildMesh();
}

static void
callbackAdaptive(bool checked, int /* a */) {

    g_adaptive = checked;
    rebuildMesh();
}

static void
callbackSmoothCornerPatch(bool checked, int /* a */) {

    g_smoothCornerPatch = checked;
    rebuildMesh();
}

static void
callbackSingleCreasePatch(bool checked, int /* a */) {

    g_singleCreasePatch = checked;
    rebuildMesh();
}

static void
callbackInfSharpPatch(bool checked, int /* a */) {

    g_infSharpPatch = checked;
    rebuildMesh();
}

static void
callbackFVarInterp(int b) {

    typedef OpenSubdiv::Sdc::Options SdcOptions;

    switch (b) {

        case SdcOptions::FVAR_LINEAR_NONE :
            g_fvarInterp = SdcOptions::FVAR_LINEAR_NONE; break;

        case SdcOptions::FVAR_LINEAR_CORNERS_ONLY :
            g_fvarInterp = SdcOptions::FVAR_LINEAR_CORNERS_ONLY; break;

        case SdcOptions::FVAR_LINEAR_CORNERS_PLUS1 :
            g_fvarInterp = SdcOptions::FVAR_LINEAR_CORNERS_PLUS1; break;

        case SdcOptions::FVAR_LINEAR_CORNERS_PLUS2 :
            g_fvarInterp = SdcOptions::FVAR_LINEAR_CORNERS_PLUS2; break;

        case SdcOptions::FVAR_LINEAR_BOUNDARIES :
            g_fvarInterp = SdcOptions::FVAR_LINEAR_BOUNDARIES; break;

        case SdcOptions::FVAR_LINEAR_ALL :
            g_fvarInterp = SdcOptions::FVAR_LINEAR_ALL; break;

    }
    rebuildMesh();
}

static void
initHUD() {

    int windowWidth = g_width, windowHeight = g_height,
        frameBufferWidth = g_width, frameBufferHeight = g_height;

    // window size might not match framebuffer size on a high DPI display
    glfwGetWindowSize(g_window, &windowWidth, &windowHeight);

    g_hud.Init(windowWidth, windowHeight, frameBufferWidth, frameBufferHeight);

    g_hud.AddCheckBox("Control edges (H)", g_controlMeshDisplay.GetEdgesDisplay(),
                      10,  60, callbackControlEdges, 0, 'h');
    g_hud.AddCheckBox("Control vertices (J)", g_controlMeshDisplay.GetVerticesDisplay(),
                      10,  80, callbackControlVertices, 0, 'j');
    g_hud.AddCheckBox("UV Backface Culling (B)", g_uvCullBackface != 0,
                      10, 100, callbackUVCullBackface, 0, 'b');

    int shading_pulldown = g_hud.AddPullDown("Display Style (W)",
                                             400, 10, 250, callbackDisplayStyle, 'w');
    g_hud.AddPullDownButton(shading_pulldown, "Wire", kWire, g_displayStyle==kWire);
    g_hud.AddPullDownButton(shading_pulldown, "Shaded", kShaded, g_displayStyle==kShaded);
    g_hud.AddPullDownButton(shading_pulldown, "Wire+Shaded", kWireShaded, g_displayStyle==kWireShaded);

    if (GLUtils::SupportsAdaptiveTessellation()) {
        g_hud.AddCheckBox("Adaptive (`)", g_adaptive != 0,
                          10, 140, callbackAdaptive, 0, '`');

        g_hud.AddCheckBox("Smooth Corner Patch (O)", g_smoothCornerPatch!=0,
                          10, 160, callbackSmoothCornerPatch, 0, 'o');
        g_hud.AddCheckBox("Single Crease Patch (S)", g_singleCreasePatch!=0,
                          10, 180, callbackSingleCreasePatch, 0, 's');
        g_hud.AddCheckBox("Inf Sharp Patch (I)", g_infSharpPatch!=0,
                          10, 200, callbackInfSharpPatch, 0, 'i');

        int endcap_pulldown = g_hud.AddPullDown("End cap (E)",
                                                10, 220, 200, callbackEndCap, 'e');
        g_hud.AddPullDownButton(endcap_pulldown, "Linear", kEndCapBilinearBasis,
                                g_endCap == kEndCapBilinearBasis);
        g_hud.AddPullDownButton(endcap_pulldown, "Regular", kEndCapBSplineBasis,
                                g_endCap == kEndCapBSplineBasis);
        g_hud.AddPullDownButton(endcap_pulldown, "Gregory", kEndCapGregoryBasis,
                                g_endCap == kEndCapGregoryBasis);
    }

    for (int i = 1; i < 11; ++i) {
        char level[16];
        sprintf(level, "Lv. %d", i);
        g_hud.AddRadioButton(3, level, i == g_level, 10, 260 + i*20, callbackLevel, i, '0'+(i%10));
    }

    typedef OpenSubdiv::Sdc::Options SdcOptions;

    int fvar_interp_pulldown = g_hud.AddPullDown("Linear Interpolation (L)",
                                                 10, 10, 250, callbackFVarInterp, 'l');
    g_hud.AddPullDownButton(fvar_interp_pulldown, "None (edge only)",
        SdcOptions::FVAR_LINEAR_NONE, g_fvarInterp==SdcOptions::FVAR_LINEAR_NONE);
    g_hud.AddPullDownButton(fvar_interp_pulldown, "Corners Only",
        SdcOptions::FVAR_LINEAR_CORNERS_ONLY, g_fvarInterp==SdcOptions::FVAR_LINEAR_CORNERS_ONLY);
    g_hud.AddPullDownButton(fvar_interp_pulldown, "Corners 1 (edge corner)",
        SdcOptions::FVAR_LINEAR_CORNERS_PLUS1, g_fvarInterp==SdcOptions::FVAR_LINEAR_CORNERS_PLUS1);
    g_hud.AddPullDownButton(fvar_interp_pulldown, "Corners 2 (edge corner prop)",
        SdcOptions::FVAR_LINEAR_CORNERS_PLUS2, g_fvarInterp==SdcOptions::FVAR_LINEAR_CORNERS_PLUS2);
    g_hud.AddPullDownButton(fvar_interp_pulldown, "Boundaries (always sharp)",
        SdcOptions::FVAR_LINEAR_BOUNDARIES, g_fvarInterp==SdcOptions::FVAR_LINEAR_BOUNDARIES);
    g_hud.AddPullDownButton(fvar_interp_pulldown, "All (bilinear)",
        SdcOptions::FVAR_LINEAR_ALL, g_fvarInterp==SdcOptions::FVAR_LINEAR_ALL);

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

    glGenVertexArrays(1, &g_vao);
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
    g_adaptive = args.GetAdaptive();
    g_level = args.GetLevel();
    g_repeatCount = args.GetRepeatCount();

    ViewerArgsUtils::PopulateShapes(args, &g_defaultShapes);

    initShapes();

    OpenSubdiv::Far::SetErrorCallback(callbackError);

    glfwSetErrorCallback(callbackErrorGLFW);
    if (! glfwInit()) {
        printf("Failed to initialize GLFW\n");
        return 1;
    }

    static const char windowTitle[] = "OpenSubdiv glFVarViewer " OPENSUBDIV_VERSION_STRING;

    GLUtils::SetMinimumGLVersion();

    if (args.GetFullScreen()) {
        g_primary = glfwGetPrimaryMonitor();

        // apparently glfwGetPrimaryMonitor fails under linux : if no primary,
        // settle for the first one in the list
        if (! g_primary) {
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

    g_adaptive = g_adaptive && GLUtils::SupportsAdaptiveTessellation();

    initGL();
    linkDefaultProgram();

    glfwSwapInterval(0);

    initHUD();
    rebuildMesh();

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
