//
//   Copyright 2015 Pixar
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
    #if defined(_WIN32)
        // XXX Must include windows.h here or GLFW pollutes the global namespace
        #define WIN32_LEAN_AND_MEAN
        #include <windows.h>
    #endif
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <GLFW/glfw3.h>

#include <osd/cpuGLVertexBuffer.h>
#include <osd/cpuComputeContext.h>
#include <osd/cpuComputeController.h>
    OpenSubdiv::Osd::CpuComputeController *g_cpuComputeController = NULL;

#ifdef OPENSUBDIV_HAS_OPENMP
    #include <osd/ompComputeController.h>
    OpenSubdiv::Osd::OmpComputeController *g_ompComputeController = NULL;
#endif

#ifdef OPENSUBDIV_HAS_TBB
    #include <osd/tbbComputeController.h>
    OpenSubdiv::Osd::TbbComputeController *g_tbbComputeController = NULL;
#endif

#ifdef OPENSUBDIV_HAS_OPENCL
    #include <osd/clGLVertexBuffer.h>
    #include <osd/clComputeContext.h>
    #include <osd/clComputeController.h>

    #include "../common/clInit.h"

    cl_context g_clContext;
    cl_command_queue g_clQueue;
    OpenSubdiv::Osd::CLComputeController *g_clComputeController = NULL;
#endif

#ifdef OPENSUBDIV_HAS_CUDA
    #include <osd/cudaGLVertexBuffer.h>
    #include <osd/cudaComputeContext.h>
    #include <osd/cudaComputeController.h>

    #include <cuda_runtime_api.h>
    #include <cuda_gl_interop.h>

    #include "../common/cudaInit.h"

    OpenSubdiv::Osd::CudaComputeController *g_cudaComputeController = NULL;
#endif

#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    #include <osd/glslTransformFeedbackComputeContext.h>
    #include <osd/glslTransformFeedbackComputeController.h>
    #include <osd/glVertexBuffer.h>
    OpenSubdiv::Osd::GLSLTransformFeedbackComputeController *g_glslTransformFeedbackComputeController = NULL;
#endif

#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    #include <osd/glslComputeContext.h>
    #include <osd/glslComputeController.h>
    #include <osd/glVertexBuffer.h>
    OpenSubdiv::Osd::GLSLComputeController *g_glslComputeController = NULL;
#endif

#include <osd/glDrawContext.h>
#include <osd/glDrawRegistry.h>
#include <osd/glMesh.h>

#include <common/vtr_utils.h>
#include "../common/patchColors.h"
#include "init_shapes.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION 1
#include "stb_image_write.h"

using namespace OpenSubdiv;

static const char *shaderSource =
#include "shader.gen.h"
;

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
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 4);
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 1);
#endif
    glfwOpenWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
}


// ---------------------------------------------------------------------------

class DrawRegistry : public Osd::GLDrawRegistry<> {
public:
    DrawRegistry(std::string const &displayMode)
        : _displayMode(displayMode) { }

protected:
    virtual SourceConfigType *
    _CreateDrawSourceConfig(DescType const & desc);

    virtual ConfigType *
    _CreateDrawConfig(DescType const & desc,
                      SourceConfigType const * sconfig);

private:
    std::string _displayMode;
};

DrawRegistry::ConfigType *
DrawRegistry::_CreateDrawConfig(DescType const & desc,
                                SourceConfigType const * sconfig) {
    ConfigType * config = BaseRegistry::_CreateDrawConfig(desc, sconfig);
    assert(config);

    GLuint uboIndex = glGetUniformBlockIndex(config->program, "Transform");
    glUniformBlockBinding(config->program, uboIndex, 0);

    GLint loc = glGetUniformLocation(config->program, "OsdPatchParamBuffer");
    if (loc != -1) {
        glUniform1i(loc, 0); // GL_TEXTURE0
    }

    return config;

};

DrawRegistry::SourceConfigType *
DrawRegistry::_CreateDrawSourceConfig(DescType const & desc) {
    typedef Far::PatchDescriptor Descriptor;

    SourceConfigType * sconfig =
        BaseRegistry::_CreateDrawSourceConfig(desc);

    assert(sconfig);

    const char *glslVersion = "#version 410\n";
    if (desc.GetType() == Descriptor::QUADS or
        desc.GetType() == Descriptor::TRIANGLES) {
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

    if (desc.GetType() == Descriptor::QUADS) {
        // uniform catmark, bilinear
        sconfig->geometryShader.AddDefine("PRIM_QUAD");
        sconfig->fragmentShader.AddDefine("PRIM_QUAD");
        sconfig->commonShader.AddDefine("UNIFORM_SUBDIVISION");
    } else if (desc.GetType() == Descriptor::TRIANGLES) {
        // uniform loop
        sconfig->geometryShader.AddDefine("PRIM_TRI");
        sconfig->fragmentShader.AddDefine("PRIM_TRI");
        sconfig->commonShader.AddDefine("LOOP");
        sconfig->commonShader.AddDefine("UNIFORM_SUBDIVISION");
    } else {
        // adaptive
        sconfig->vertexShader.source =
            shaderSource + sconfig->vertexShader.source;
        sconfig->tessControlShader.source =
            shaderSource + sconfig->tessControlShader.source;
        sconfig->tessEvalShader.source =
            shaderSource + sconfig->tessEvalShader.source;

        sconfig->geometryShader.AddDefine("PRIM_TRI");
        sconfig->fragmentShader.AddDefine("PRIM_TRI");
    }

    sconfig->commonShader.AddDefine("DISPLAY_MODE_" + _displayMode);

//    sconfig->commonShader.AddDefine("OSD_ENABLE_SCREENSPACE_TESSELLATION");
//    sconfig->commonShader.AddDefine("OSD_FRACTIONAL_ODD_SPACING");
    sconfig->commonShader.AddDefine("OSD_ENABLE_PATCH_CULL");
    sconfig->commonShader.AddDefine("GEOMETRY_OUT_LINE");

    return sconfig;
}

// ---------------------------------------------------------------------------

static Osd::GLMeshInterface *
createOsdMesh(std::string const &kernel,
              Far::TopologyRefiner *refiner,
              int numVertexElements,
              int numVaryingElements,
              int level,
              Osd::MeshBitset bits)
{
    if (kernel == "CPU") {
        if (not g_cpuComputeController) {
            g_cpuComputeController = new Osd::CpuComputeController();
        }
        return new Osd::Mesh<Osd::CpuGLVertexBuffer,
            Osd::CpuComputeController,
            Osd::GLDrawContext>(
                g_cpuComputeController,
                refiner,
                numVertexElements,
                numVaryingElements,
                level, bits);
#ifdef OPENSUBDIV_HAS_OPENMP
    } else if (kernel == "OPENMP") {
        if (not g_ompComputeController) {
            g_ompComputeController = new Osd::OmpComputeController();
        }
        return new Osd::Mesh<Osd::CpuGLVertexBuffer,
            Osd::OmpComputeController,
            Osd::GLDrawContext>(
                g_ompComputeController,
                refiner,
                numVertexElements,
                numVaryingElements,
                level, bits);
#endif
#ifdef OPENSUBDIV_HAS_TBB
    } else if (kernel == "TBB") {
        if (not g_tbbComputeController) {
            g_tbbComputeController = new Osd::TbbComputeController();
        }
        return new Osd::Mesh<Osd::CpuGLVertexBuffer,
            Osd::TbbComputeController,
            Osd::GLDrawContext>(
                g_tbbComputeController,
                refiner,
                numVertexElements,
                numVaryingElements,
                level, bits);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    } else if(kernel == "CL") {
        if (not g_clComputeController) {
            g_clComputeController = new Osd::CLComputeController(g_clContext, g_clQueue);
        }
        return new Osd::Mesh<Osd::CLGLVertexBuffer,
            Osd::CLComputeController,
            Osd::GLDrawContext>(
                g_clComputeController,
                refiner,
                numVertexElements,
                numVaryingElements,
                level, bits, g_clContext, g_clQueue);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    } else if(kernel == "CUDA") {
        if (not g_cudaComputeController) {
            g_cudaComputeController = new Osd::CudaComputeController();
        }
        return new Osd::Mesh<Osd::CudaGLVertexBuffer,
            Osd::CudaComputeController,
            Osd::GLDrawContext>(
                g_cudaComputeController,
                refiner,
                numVertexElements,
                numVaryingElements,
                level, bits);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    } else if(kernel == "XFB") {
        if (not g_glslTransformFeedbackComputeController) {
            g_glslTransformFeedbackComputeController = new Osd::GLSLTransformFeedbackComputeController();
        }
        return new Osd::Mesh<Osd::GLVertexBuffer,
            Osd::GLSLTransformFeedbackComputeController,
            Osd::GLDrawContext>(
                g_glslTransformFeedbackComputeController,
                refiner,
                numVertexElements,
                numVaryingElements,
                level, bits);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    } else if(kernel == "GLSL") {
        if (not g_glslComputeController) {
            g_glslComputeController = new Osd::GLSLComputeController();
        }
        return new Osd::Mesh<Osd::GLVertexBuffer,
            Osd::GLSLComputeController,
            Osd::GLDrawContext>(
                g_glslComputeController,
                refiner,
                numVertexElements,
                numVaryingElements,
                level, bits);
#endif
    }

    std::cout << "Specified kernel is not supported in this build.\n";
    exit(1);
}

void runTest(ShapeDesc const &shapeDesc, std::string const &kernel,
             int level, bool adaptive,
             DrawRegistry *drawRegistry) {

    std::cout << "Testing " << shapeDesc.name << ", kernel = " << kernel << "\n";

    Shape const * shape = Shape::parseObj(shapeDesc.data.c_str(),
                                          shapeDesc.scheme);

    // create Vtr mesh (topology)
    Sdc::SchemeType sdctype = GetSdcType(*shape);
    Sdc::Options sdcoptions = GetSdcOptions(*shape);

    // create topology refiner
    Far::TopologyRefiner *refiner =
        Far::TopologyRefinerFactory<Shape>::Create(
            *shape,
            Far::TopologyRefinerFactory<Shape>::Options(sdctype, sdcoptions));

    // Adaptive refinement currently supported only for catmull-clark scheme
    bool doAdaptive = adaptive && (shapeDesc.scheme == kCatmark);
    bool interleaveVarying = true;
    bool doSingleCreasePatch = true;

    Osd::MeshBitset bits;
    bits.set(Osd::MeshAdaptive, doAdaptive);
    bits.set(Osd::MeshUseSingleCreasePatch, doSingleCreasePatch);
    bits.set(Osd::MeshInterleaveVarying, interleaveVarying);
    bits.set(Osd::MeshFVarData, false);
    bits.set(Osd::MeshUseGregoryBasis, true);

    int numVertexElements = 3 + 4; // XYZ, RGBA (interleaved)
    int numVaryingElements = 0;

    Osd::GLMeshInterface *mesh = createOsdMesh(
        kernel, refiner, numVertexElements, numVaryingElements, level, bits);

    int nverts = shape->GetNumVertices();
    // centering
    float pmin[3] = { std::numeric_limits<float>::max(),
                      std::numeric_limits<float>::max(),
                      std::numeric_limits<float>::max() };
    float pmax[3] = { -std::numeric_limits<float>::max(),
                      -std::numeric_limits<float>::max(),
                      -std::numeric_limits<float>::max() };
    for (int i = 0; i < nverts; ++i) {
        for (int j = 0; j < 3; ++j) {
            float v = shape->verts[i*3+j];
            pmin[j] = std::min(v, pmin[j]);
            pmax[j] = std::max(v, pmax[j]);
        }
    }
    float center[3] = { (pmax[0]+pmin[0])*0.5f,
                        (pmax[1]+pmin[1])*0.5f,
                        (pmax[2]+pmin[2])*0.5f };
    float radius = sqrt((pmax[0]-pmin[0])*(pmax[0]-pmin[0]) +
                        (pmax[1]-pmin[1])*(pmax[1]-pmin[1]) +
                        (pmax[2]-pmin[2])*(pmax[2]-pmin[2]));

    // prepare coarse vertices
    std::vector<float> vertex;
    vertex.resize(nverts * numVertexElements);
    for (int i = 0; i < nverts; ++i) {
        for (int j = 0; j < 3; ++j) {
            vertex[i*numVertexElements+j] =
                (shape->verts[i*3+j] - center[j])/radius;
        }
        for (int j = 0; j < 4; ++j) {
            vertex[i*numVertexElements+j+3] =
                (shape->verts[i*3+j] - pmin[j])*2.0f/radius;
        }
    }
    mesh->UpdateVertexBuffer(&vertex[0], 0, nverts);

    // refine
    mesh->Refine();

    // draw
    glClearColor(0.1f, 0.1f, 0.1f, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // bind vertex
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh->GetDrawContext()->GetPatchIndexBuffer());
    glBindBuffer(GL_ARRAY_BUFFER, mesh->BindVertexBuffer());

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                          sizeof (GLfloat) * numVertexElements, 0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE,
                          sizeof (GLfloat) * numVertexElements,
                          (const void*)(sizeof(GLfloat)*3));

    // bind patchparam
    if (mesh->GetDrawContext()->GetPatchParamTextureBuffer()) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_BUFFER,
                      mesh->GetDrawContext()->GetPatchParamTextureBuffer());
    }

    Osd::DrawContext::PatchArrayVector const & patches =
        mesh->GetDrawContext()->GetPatchArrays();

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
            primType = GL_PATCHES;
            glPatchParameteri(GL_PATCH_VERTICES, desc.GetNumControlVertices());
        }

        GLuint program = drawRegistry->GetDrawConfig(desc)->program;
        glUseProgram(program);

        GLuint diffuseColor =
            glGetUniformLocation(program, "diffuseColor");
        GLuint uniformPrimitiveIdBase =
            glGetUniformLocation(program, "PrimitiveIdBase");

        if (primType == GL_PATCHES) {
            float const * color = getAdaptivePatchColor( desc );
            glProgramUniform4f(program, diffuseColor,
                               color[0], color[1], color[2], color[3]);
            glProgramUniform1i(program, uniformPrimitiveIdBase,
                               patch.GetPatchIndex());
        } else {
            glProgramUniform4f(program, diffuseColor, 0.4f, 0.4f, 0.8f, 1);
        }

        glDrawElements(primType, patch.GetNumIndices(), GL_UNSIGNED_INT,
                       (void *)(patch.GetVertIndex() * sizeof(unsigned int)));
    }

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

    glBindVertexArray(0);
    glDeleteVertexArrays(1, &vao);

    // cleanup
    delete shape;
    delete mesh;

    // mesh takes an ownership of topologyRefiner. no need to delete it.
}

static void usage(const char *program) {
    std::cout
        << "Usage %s : " << program << "\n"
        << "   -a                      : adaptive refinement\n"
        << "   -l <isolation level>    : isolation level (default = 2)\n"
        << "   -t <tess level>         : tessellation level (default = 1)\n"
        << "   -w <prefix>             : write images to PNG as\n"
        << "                             <prefix>_<kernel>_modelname.png\n"
        << "   -s <width> <height>     : image size (default = 128 128)\n"
        << "   -k <kernel>,<kernel>... : kernel types (default = all)\n"
        << "      kernel = [CPU, OPENMP, TBB, CUDA, CL, XFB, GLSL]\n"
        << "   -d <displayMode>        : display mode\n"
        << "      displayMode = [PATCH_TYPE, VARYING, NORMAL]\n";
}

int main(int argc, char ** argv) {

    int width = 128;
    int height = 128;
    int tessLevel = 1;
    int isolationLevel = 2;
    bool writeToFile = false;
    bool adaptive = false;
    std::string prefix;
    std::string displayMode = "PATCH_TYPE";
    std::vector<std::string> kernels;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-a")) {
            adaptive = true;
        } else if (!strcmp(argv[i], "-l")) {
            isolationLevel = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-k")) {
            std::stringstream ss(argv[++i]);
            std::string kernel;
            while(std::getline(ss, kernel, ',')) {
                kernels.push_back(kernel);
            }
        } else if (!strcmp(argv[i], "-t")) {
            tessLevel = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-w")) {
            writeToFile = true;
            prefix = argv[++i];
        } else if (!strcmp(argv[i], "-s")) {
            width = atoi(argv[++i]);
            height = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-d")) {
            displayMode = argv[++i];
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    // by default, test all available kernels
    if (kernels.empty()) {
        kernels.push_back("CPU");
#ifdef OPENSUBDIV_HAS_OPENMP
        kernels.push_back("OPENMP");
#endif
#ifdef OPENSUBDIV_HAS_TBB
        kernels.push_back("TBB");
#endif
#ifdef OPENSUBDIV_HAS_CUDA
        kernels.push_back("CUDA");
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
        kernels.push_back("CL");
#endif
#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
        kernels.push_back("XFB");
#endif
#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
        kernels.push_back("GLSL");
#endif
    }

    if (not glfwInit()) {
        std::cout << "Failed to initialize GLFW\n";
        return 1;
    }

    static const char windowTitle[] =
        "OpenSubdiv imaging test " OPENSUBDIV_VERSION_STRING;

    setGLCoreProfile();

    GLFWwindow *window = glfwCreateWindow(width, height, windowTitle, NULL, NULL);
    if (not window) {
        std::cout << "Failed to open window.\n";
        glfwTerminate();
    }
    glfwMakeContextCurrent(window);

#if defined(OSD_USES_GLEW)
    // this is the only way to initialize glew correctly under core profile context.
    glewExperimental = true;
    if (GLenum r = glewInit() != GLEW_OK) {
        std::cout << "Failed to initialize glew. Error = "
                  << glewGetErrorString(r) << "\n";
        exit(1);
    }
    // clear GL errors generated during glewInit()
    glGetError();
#endif

    initShapes();

    // initialize GL states
    glViewport(0, 0, width, height);
    glEnable(GL_DEPTH_TEST);

    // some regression shapes are not visible in this camera
    // with backface culling.
    // glEnable(GL_CULL_FACE);

    // transform uniform
    float modelview[16] = {
        0.945518f, -0.191364f,  0.263390f, 0.000000f,
        0.325568f,  0.555762f, -0.764941f, 0.000000f,
        0.000000f,  0.809017f,  0.587785f, 0.000000f,
        0.000000f,  0.000000f, -1.500000f, 1.000000f };
    float projection[16] = {
        2.414213f, 0.000000f,  0.000000f,  0.000000f,
        0.000000f, 2.414213f,  0.000000f,  0.000000f,
        0.000000f, 0.000000f, -1.000000f, -1.000000f,
        0.000000f, 0.000000f, -0.000200f,  0.000000f
    };

    struct Transform {
        float ModelViewMatrix[16];
        float ProjectionMatrix[16];
        float Viewport[4];
        float TessLevel;
    } transformData;

    memcpy(transformData.ModelViewMatrix, modelview, sizeof(modelview));
    memcpy(transformData.ProjectionMatrix, projection, sizeof(projection));
    transformData.Viewport[0] = 0;
    transformData.Viewport[1] = 0;
    transformData.Viewport[2] = static_cast<float>(width);
    transformData.Viewport[3] = static_cast<float>(height);
    transformData.TessLevel = static_cast<float>(1 << tessLevel);

    GLuint transformUB = 0;
    glGenBuffers(1, &transformUB);
    glBindBuffer(GL_UNIFORM_BUFFER, transformUB);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(transformData),
                 &transformData, GL_STATIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, /*binding=*/0, transformUB);

    // create draw registry;
    DrawRegistry drawRegistry(displayMode);

    // write report html
    if (writeToFile) {
        std::string reportfile = prefix + ".html";
        std::ofstream ofs(reportfile.c_str());

        ofs << "<html>\n"
            << "<head><style>\n"
            << "table { border-collapse:collapse; } "
            << "table,th,td {border: 1px solid black} "
            << "</style></head>\n";

        ofs << "<body>\n";
        ofs << "<h3>OpenSubdiv imaging regression test<h3>\n";
        ofs << "<pre>\n";
        ofs << "OpenSubdiv      : " << OPENSUBDIV_VERSION_STRING << "\n";
        ofs << "GL Version      : " << glGetString(GL_VERSION)
            << ", " << glGetString(GL_VENDOR)
            << ", " << glGetString(GL_RENDERER)
            << "\n";
        ofs << "Isolation Level : " << isolationLevel << "\n";
        ofs << "Tess Level      : " << tessLevel << "\n";
        ofs << "Adaptive        : " << adaptive << "\n";
        ofs << "Display Mode    : " << displayMode << "\n";
        ofs << "</pre>\n";

        ofs << "<table>\n";
        ofs << "<tr>\n";
        ofs << "<th>Reference<br>(on github. to be updated)</th>\n";
        for (size_t k = 0; k < kernels.size(); ++k) {
            ofs << "<th>" << kernels[k] << "</th>\n";
        }
        ofs << "</tr>\n";

        for (size_t i = 0; i < g_shapes.size(); ++i) {
            ofs << "<tr>\n";
            ofs << "<td>" << g_shapes[i].name << "</td>\n";
            for (size_t k = 0; k < kernels.size(); ++k) {
                ofs << "<td>";
                ofs << "<img src='" << prefix << "_" << kernels[k] << "_" << g_shapes[i].name << ".png'>";
                ofs << "</td>";
            }
            ofs << "</tr>\n";
        }
        ofs << "</table>\n";
        ofs << "</body></html>\n";
        ofs.close();
    }

    // run test
    for (size_t k = 0; k < kernels.size(); ++k) {
        std::string const &kernel = kernels[k];

        // prep GPU kernel
#ifdef OPENSUBDIV_HAS_OPENCL
        if (kernel == "CL") {
            if (initCL(&g_clContext, &g_clQueue) == false) {
                std::cout << "Error in initializing OpenCL\n";
                exit(1);
            }
        }
#endif
#ifdef OPENSUBDIV_HAS_CUDA
        if (kernel == "CUDA") {
            cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
        }
#endif
        for (size_t i = 0; i < g_shapes.size(); ++i) {
            // run test
            runTest(g_shapes[i], kernel, isolationLevel, adaptive, &drawRegistry);

            if (writeToFile) {
                // read back pixels
                std::vector<unsigned char> data(width*height*3);
                glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, &data[0]);

                // write image
                std::string filename = prefix + "_" + kernel + "_" + g_shapes[i].name + ".png";
                // flip vertical
                stbi_write_png(filename.c_str(), width, height, 3, &data[width*3*(height-1)], -width*3);
            }

            glfwSwapBuffers(window);
        }

#ifdef OPENSUBDIV_HAS_OPENCL
        if (kernel == "CL") {
            uninitCL(g_clContext, g_clQueue);
        }
#endif
    }

    return 0;
}
