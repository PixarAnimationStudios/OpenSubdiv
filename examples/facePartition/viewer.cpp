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

#if defined(GLFW_VERSION_3)
    #include <GLFW/glfw3.h>
    GLFWwindow* g_window=0;
    GLFWmonitor* g_primary=0;
#else
    #include <GL/glfw.h>
#endif

#include <far/mesh.h>
#include <far/meshFactory.h>

#include <osd/error.h>
#include <osd/vertex.h>
#include <osd/glDrawContext.h>
#include <osd/glDrawRegistry.h>

#include <osd/cpuGLVertexBuffer.h>
#include <osd/cpuComputeContext.h>
#include <osd/cpuComputeController.h>

#include <osdutil/patchPartitioner.h>

OpenSubdiv::OsdCpuComputeController *g_cpuComputeController = NULL;

class PartitionedMesh {
public:
    PartitionedMesh(OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex> const *farMesh,
                    std::vector<int> const &partitionPerFace) {

        int numVertices = farMesh->GetNumVertices();
        _vertexBuffer = OpenSubdiv::OsdCpuGLVertexBuffer::Create(3, numVertices);
        _computeContext = OpenSubdiv::OsdCpuComputeContext::Create(farMesh);
        _kernelBatches = farMesh->GetKernelBatches();

        OpenSubdiv::OsdUtilPatchPartitioner partitioner(farMesh->GetPatchTables(), partitionPerFace);

        // convert farpatch to osdpatch
        int maxMaterial = partitioner.GetNumPartitions();
        int maxValence = farMesh->GetPatchTables()->GetMaxValence();;

        _partitionedOsdPatchArrays.resize(maxMaterial);
        for (int i = 0; i < maxMaterial; ++i) {
            OpenSubdiv::OsdDrawContext::ConvertPatchArrays(partitioner.GetPatchArrays(i),
                                                           _partitionedOsdPatchArrays[i],
                                                           maxValence, 3);
        }

        _drawContext = OpenSubdiv::OsdGLDrawContext::Create(&partitioner.GetPatchTables(), false);
        _drawContext->UpdateVertexTexture(_vertexBuffer);
    }

    ~PartitionedMesh() {
        delete _vertexBuffer;
        delete _computeContext;
        delete _drawContext;
    }

    void UpdateVertexBuffer(float const *vertexData, int numVerts) {
        _vertexBuffer->UpdateData(vertexData, 0, numVerts);
    }

    void Refine() {
        g_cpuComputeController->Refine(_computeContext,
                                       _kernelBatches,
                                       _vertexBuffer);
  }

  OpenSubdiv::OsdGLDrawContext *GetDrawContext() const {
    return _drawContext;
  }
  GLuint BindVertexBuffer() {
    return _vertexBuffer->BindVBO();
  }

  int GetNumPartitions() const {
    return (int)_partitionedOsdPatchArrays.size();
  }

  OpenSubdiv::OsdDrawContext::PatchArrayVector const & GetPatchArrays(int partition) const {
    return _partitionedOsdPatchArrays[partition];
  }

private:

  OpenSubdiv::OsdCpuComputeContext *_computeContext;
  OpenSubdiv::OsdCpuGLVertexBuffer *_vertexBuffer;
  OpenSubdiv::FarKernelBatchVector _kernelBatches;

  OpenSubdiv::OsdGLDrawContext *_drawContext;
  std::vector<OpenSubdiv::OsdDrawContext::PatchArrayVector> _partitionedOsdPatchArrays;
};

PartitionedMesh *g_mesh = NULL;

#include <common/shape_utils.h>
#include "../common/stopwatch.h"
#include "../common/simple_math.h"
#include "../common/gl_hud.h"

static const char *shaderSource =
#include "shader.gen.h"
;

#include <cfloat>
#include <vector>
#include <fstream>
#include <sstream>

typedef OpenSubdiv::HbrMesh<OpenSubdiv::OsdVertex>     OsdHbrMesh;
typedef OpenSubdiv::HbrVertex<OpenSubdiv::OsdVertex>   OsdHbrVertex;
typedef OpenSubdiv::HbrFace<OpenSubdiv::OsdVertex>     OsdHbrFace;
typedef OpenSubdiv::HbrHalfedge<OpenSubdiv::OsdVertex> OsdHbrHalfedge;

enum DisplayStyle { kWire = 0,
                    kShaded,
                    kWireShaded };

enum HudCheckBox { kHUD_CB_PARTITIONING };

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

// GUI variables
int   g_displayStyle = kWireShaded,
      g_adaptive = 0,
      g_mbutton[3] = {0, 0, 0},
      g_partitioning = 1,
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

// geometry
std::vector<float> g_orgPositions,
                   g_positions;

Scheme             g_scheme;

int g_level = 2;
int g_tessLevel = 1;
int g_tessLevelMin = 1;

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

GLuint g_primQuery = 0;
GLuint g_vao = 0;

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

#include <shapes/catmark_chaikin0.h>
    g_defaultShapes.push_back(SimpleShape(catmark_chaikin0, "catmark_chaikin0", kCatmark));

#include <shapes/catmark_chaikin1.h>
    g_defaultShapes.push_back(SimpleShape(catmark_chaikin1, "catmark_chaikin1", kCatmark));

#include <shapes/catmark_fan.h>
    g_defaultShapes.push_back(SimpleShape(catmark_fan, "catmark_fan", kCatmark));

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

#include <shapes/loop_chaikin0.h>
    g_defaultShapes.push_back(SimpleShape(loop_chaikin0, "loop_chaikin0", kLoop));

#include <shapes/loop_chaikin1.h>
    g_defaultShapes.push_back(SimpleShape(loop_chaikin1, "loop_chaikin1", kLoop));
}

//------------------------------------------------------------------------------
static void
updateGeom() {

    int nverts = (int)g_orgPositions.size() / 3;

    std::vector<float> vertex;
    vertex.reserve(nverts*3);

    const float *p = &g_orgPositions[0];

    for (int i = 0; i < nverts; ++i) {
        g_positions[i*3+0] = p[0];
        g_positions[i*3+1] = p[1];
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

    g_mesh->UpdateVertexBuffer(&vertex[0], nverts);

    Stopwatch s;
    s.Start();

    g_mesh->Refine();
}

//------------------------------------------------------------------------------
static void
createOsdMesh( const std::string &shape, int level, Scheme scheme=kCatmark ) {

    checkGLErrors("create osd enter");
    // generate Hbr representation from "obj" description
    OsdHbrMesh * hmesh = simpleHbr<OpenSubdiv::OsdVertex>(shape.c_str(), scheme, g_orgPositions);

    // material assignment
    std::vector<int> idsOnPtexFaces;
    {
        int numFaces = hmesh->GetNumCoarseFaces();

        // first, assign material ID to each coarse face
        std::vector<int> idsOnCoarseFaces;
        for (int i = 0; i < numFaces; ++i) {
            int materialID = i%6;
            idsOnCoarseFaces.push_back(materialID);
        }

        // create ptex index to coarse face index mapping
        OsdHbrFace *lastFace = hmesh->GetFace(numFaces-1);
        int numPtexFaces = lastFace->GetPtexIndex();
        numPtexFaces += (hmesh->GetSubdivision()->FaceIsExtraordinary(hmesh, lastFace) ?
                         lastFace->GetNumVertices() : 1);

        // XXX: duped logic to simpleHbr
        std::vector<int> ptexIndexToFaceMapping(numPtexFaces);
        int ptexIndex = 0;
        for (int i = 0; i < numFaces; ++i) {
            OsdHbrFace * f = hmesh->GetFace(i);
            ptexIndexToFaceMapping[ptexIndex++] = i;
            int numVerts = f->GetNumVertices();
            if ( (scheme==kCatmark or scheme==kBilinear) and numVerts != 4 ) {
                for (int j = 0; j < numVerts-1; ++j) {
                    ptexIndexToFaceMapping[ptexIndex++] = i;
                }
            }
        }
        assert((int)ptexIndexToFaceMapping.size() == numPtexFaces);

        // convert ID array from coarse face index space to ptex index space
        for (int i = 0; i < numPtexFaces; ++i) {
            idsOnPtexFaces.push_back(idsOnCoarseFaces[ptexIndexToFaceMapping[i]]);
        }
    }

    // Adaptive refinement currently supported only for catmull-clark scheme
    g_scheme = scheme;
    bool doAdaptive = (g_adaptive!=0 and g_scheme==kCatmark);

    // create farmesh
    OpenSubdiv::FarMeshFactory<OpenSubdiv::OsdVertex> meshFactory(hmesh, level, doAdaptive);
    OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex> *farMesh = meshFactory.Create();

    // create partitioned patcharray
    delete g_mesh;
    g_mesh = new PartitionedMesh(farMesh, idsOnPtexFaces);

    // Hbr,Far mesh can be deleted
    delete hmesh;
    delete farMesh;

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

    g_positions.resize(g_orgPositions.size(),0.0f);

    g_tessLevelMin = 1;

    g_tessLevel = std::max(g_tessLevel,g_tessLevelMin);

    updateGeom();

    // -------- VAO
    glBindVertexArray(g_vao);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_mesh->GetDrawContext()->GetPatchIndexBuffer());
    glBindBuffer(GL_ARRAY_BUFFER, g_mesh->BindVertexBuffer());

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 3, 0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
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

typedef std::pair<OpenSubdiv::OsdDrawContext::PatchDescriptor, Effect> EffectDesc;

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
GetEffect()
{
    return Effect(g_displayStyle);
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
    if (g_mesh->GetDrawContext()->GetFvarDataTextureBuffer()) {
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_BUFFER,
            g_mesh->GetDrawContext()->GetFvarDataTextureBuffer());
    }

    glActiveTexture(GL_TEXTURE0);

    return program;
}

//------------------------------------------------------------------------------
static int
drawPatches(OpenSubdiv::OsdDrawContext::PatchArrayVector const &patches, GLfloat const *color)
{
    int numDrawCalls = 0;
    for (int i=0; i<(int)patches.size(); ++i) {
        OpenSubdiv::OsdDrawContext::PatchArray const & patch = patches[i];

        OpenSubdiv::OsdDrawContext::PatchDescriptor desc = patch.GetDescriptor();
        OpenSubdiv::FarPatchTables::Type patchType = desc.GetType();

        GLenum primType;

        switch(patchType) {
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

        glDrawElements(primType, patch.GetNumIndices(), GL_UNSIGNED_INT,
                       (void *)(patch.GetVertIndex() * sizeof(unsigned int)));
        ++numDrawCalls;
    }
    return numDrawCalls;
}
static void
display() {

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

    // make sure that the vertex buffer is interoped back as a GL resources.
    g_mesh->BindVertexBuffer();

    glBindVertexArray(g_vao);

    // patch drawing
    int numDrawCalls = 0;

    // primitive counting
    glBeginQuery(GL_PRIMITIVES_GENERATED, g_primQuery);

    if (g_partitioning) {
        // draw for each partition
        static GLfloat color[][3] = { {1, 0, 0},
                                      {0, 1, 0},
                                      {0, 0, 1},
                                      {1, 1, 0},
                                      {1, 0, 1},
                                      {0, 1, 1} };
        for (int material = 0; material < g_mesh->GetNumPartitions(); ++material) {
            OpenSubdiv::OsdDrawContext::PatchArrayVector const & patches = g_mesh->GetPatchArrays(material);

            numDrawCalls += drawPatches(patches, color[material]);
        }
    } else {
        // draw at once
        static GLfloat color[3] = {0.5, 0.5, 0.5};
        OpenSubdiv::OsdDrawContext::PatchArrayVector const & patches = g_mesh->GetDrawContext()->patchArrays;
        numDrawCalls += drawPatches(patches, color);
    }

    glEndQuery(GL_PRIMITIVES_GENERATED);

    GLuint numPrimsGenerated = 0;
    glGetQueryObjectuiv(g_primQuery, GL_QUERY_RESULT, &numPrimsGenerated);

    glBindVertexArray(0);

    glUseProgram(0);

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

        g_hud.DrawString(10, -180, "Tess level : %d", g_tessLevel);
        g_hud.DrawString(10, -160, "Primitives : %d", numPrimsGenerated);
        g_hud.DrawString(10, -140, "Draw calls : %d", numDrawCalls);
        g_hud.DrawString(10, -120, "Scheme     : %s", g_scheme==kBilinear ? "BILINEAR" : (g_scheme == kLoop ? "LOOP" : "CATMARK"));
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
mouse(GLFWwindow *, int button, int state, int mods) {
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
uninitGL() {

    glDeleteQueries(1, &g_primQuery);
    glDeleteVertexArrays(1, &g_vao);

    if (g_mesh)
        delete g_mesh;

    delete g_cpuComputeController;
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

    int windowWidth = g_width, windowHeight = g_height;
#if GLFW_VERSION_MAJOR>=3
    // window size might not match framebuffer size on a high DPI display
    glfwGetWindowSize(g_window, &windowWidth, &windowHeight);
#endif
    g_hud.Rebuild(windowWidth, windowHeight);
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
    createOsdMesh( g_defaultShapes[ g_currentShape ].data, g_level, g_defaultShapes[ g_currentShape ].scheme );
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
    if (m < 0)
        m = 0;

    if (m >= (int)g_defaultShapes.size())
        m = (int)g_defaultShapes.size() - 1;

    g_currentShape = m;
    rebuildOsdMesh();
}

static void
callbackDisplayStyle(int b)
{
    g_displayStyle = b;
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
callbackCheckBox(bool checked, int button)
{
    switch (button) {
    case kHUD_CB_PARTITIONING:
        g_partitioning = checked;
        break;
    }
}

static void
initHUD()
{
    int windowWidth = g_width, windowHeight = g_height;
#if GLFW_VERSION_MAJOR>=3
    // window size might not match framebuffer size on a high DPI display
    glfwGetWindowSize(g_window, &windowWidth, &windowHeight);
#endif
    g_hud.Init(windowWidth, windowHeight);

    g_hud.AddRadioButton(1, "Wire (W)",    g_displayStyle == kWire,  200, 10, callbackDisplayStyle, 0, 'w');
    g_hud.AddRadioButton(1, "Shaded",      g_displayStyle == kShaded, 200, 30, callbackDisplayStyle, 1, 'w');
    g_hud.AddRadioButton(1, "Wire+Shaded", g_displayStyle == kWireShaded, 200, 50, callbackDisplayStyle, 2, 'w');

    g_hud.AddCheckBox("Partitioning", g_partitioning != 0,
                      350, 10, callbackCheckBox, kHUD_CB_PARTITIONING, 'p');


    if (OpenSubdiv::OsdGLDrawContext::SupportsAdaptiveTessellation())
        g_hud.AddCheckBox("Adaptive (`)", g_adaptive!=0, 10, 190, callbackAdaptive, 0, '`');

    for (int i = 1; i < 11; ++i) {
        char level[16];
        sprintf(level, "Lv. %d", i);
        g_hud.AddRadioButton(3, level, i==2, 10, 210+i*20, callbackLevel, i, '0'+(i%10));
    }

    for (int i = 0; i < (int)g_defaultShapes.size(); ++i) {
        g_hud.AddRadioButton(4, g_defaultShapes[i].name.c_str(), i==g_currentShape, -220, 10+i*16, callbackModel, i, 'n');
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

    glGenQueries(1, &g_primQuery);

    glGenVertexArrays(1, &g_vao);
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
int main(int argc, char ** argv)
{
    std::string str;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-d"))
            g_level = atoi(argv[++i]);
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

    static const char windowTitle[] = "OpenSubdiv face partitioning example";

#define CORE_PROFILE
#ifdef CORE_PROFILE
    setGLCoreProfile();
#endif

#if GLFW_VERSION_MAJOR>=3
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
#else
    if (glfwOpenWindow(g_width, g_height, 8, 8, 8, 8, 24, 8, GLFW_WINDOW) == GL_FALSE) {
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

    // activate feature adaptive tessellation if OSD supports it
    g_adaptive = OpenSubdiv::OsdGLDrawContext::SupportsAdaptiveTessellation();

    initGL();

    glfwSwapInterval(0);

    g_cpuComputeController = new OpenSubdiv::OsdCpuComputeController();

    initHUD();
    rebuildOsdMesh();

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

//------------------------------------------------------------------------------
