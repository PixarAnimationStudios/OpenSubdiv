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

#include "glLoader.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <GLFW/glfw3.h>

#include <opensubdiv/osd/cpuEvaluator.h>
#include <opensubdiv/osd/cpuGLVertexBuffer.h>

#ifdef OPENSUBDIV_HAS_OPENMP
    #include <opensubdiv/osd/ompEvaluator.h>
#endif

#ifdef OPENSUBDIV_HAS_TBB
    #include <opensubdiv/osd/tbbEvaluator.h>
#endif

#ifdef OPENSUBDIV_HAS_OPENCL
    #include <opensubdiv/osd/clEvaluator.h>
    #include <opensubdiv/osd/clGLVertexBuffer.h>

    #include "../common/clDeviceContext.h"
    CLDeviceContext g_clDeviceContext;
#endif

#ifdef OPENSUBDIV_HAS_CUDA
    #include <opensubdiv/osd/cudaEvaluator.h>
    #include <opensubdiv/osd/cudaGLVertexBuffer.h>
    #include "../common/cudaDeviceContext.h"
    CudaDeviceContext g_cudaDeviceContext;
#endif

#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    #include <opensubdiv/osd/glXFBEvaluator.h>
    #include <opensubdiv/osd/glVertexBuffer.h>
#endif

#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    #include <opensubdiv/osd/glComputeEvaluator.h>
    #include <opensubdiv/osd/glVertexBuffer.h>
#endif

#include <opensubdiv/osd/glMesh.h>

#include "../../regression/common/far_utils.h"
#include "../../regression/common/arg_utils.h"
#include "../common/patchColors.h"
#include "../common/stb_image_write.h"    // common.obj has an implementation.
#include "../common/glShaderCache.h"
#include "../common/glUtils.h"
#include "init_shapes.h"

using namespace OpenSubdiv;

#include <opensubdiv/osd/glslPatchShaderSource.h>
static const char *shaderSource =
#include "shader.gen.h"
;

class ShaderCache : public GLShaderCache<OpenSubdiv::Far::PatchDescriptor> {
public:
    ShaderCache(std::string const &displayMode) : _displayMode(displayMode) { }

    virtual GLDrawConfig *CreateDrawConfig(OpenSubdiv::Far::PatchDescriptor const &desc) {

        using namespace OpenSubdiv;

        // compile shader program
        GLDrawConfig *config =
            new GLDrawConfig(GLUtils::GetShaderVersionInclude().c_str());

        Far::PatchDescriptor::Type type = desc.GetType();

        // common defines
        std::stringstream ss;

        if (type == Far::PatchDescriptor::QUADS) {
            ss << "#define PRIM_QUAD\n";
        } else {
            ss << "#define PRIM_TRI\n";
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
        
        if (desc.IsAdaptive()) {
            ss << "#define SMOOTH_NORMALS\n";
        }
        ss << "#define DISPLAY_MODE_" <<  _displayMode << "\n";
        ss << "#define OSD_ENABLE_PATCH_CULL\n";
        ss << "#define GEOMETRY_OUT_LINE\n";

        if (desc.IsAdaptive() && type == Far::PatchDescriptor::REGULAR) {
            ss << "#define OSD_PATCH_ENABLE_SINGLE_CREASE\n";
        }

        // include osd PatchCommon
        ss << Osd::GLSLPatchShaderSource::GetCommonShaderSource();
        std::string common = ss.str();
        ss.str("");

        // vertex shader
        ss << common
           << (desc.IsAdaptive() ? "" : "#define VERTEX_SHADER\n") // for my shader source
           << shaderSource
           << Osd::GLSLPatchShaderSource::GetVertexShaderSource(type);
        config->CompileAndAttachShader(GL_VERTEX_SHADER, ss.str());
        ss.str("");

        if (desc.IsAdaptive()) {
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
           << "#define GEOMETRY_SHADER\n" // for my shader source
           << shaderSource;
        config->CompileAndAttachShader(GL_GEOMETRY_SHADER, ss.str());
        ss.str("");

        // fragment shader
        ss << common
           << "#define FRAGMENT_SHADER\n" // for my shader source
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
        uboIndex = glGetUniformBlockIndex(program, "Transform");
        if (uboIndex != GL_INVALID_INDEX)
            glUniformBlockBinding(program, uboIndex, 0);

        // assign texture locations
        GLint loc;
        if ((loc = glGetUniformLocation(program, "OsdPatchParamBuffer")) != -1) {
            glProgramUniform1i(program, loc, 0); // GL_TEXTURE0
        }

        return config;
    }
private:
    std::string _displayMode;
};

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
        return new Osd::Mesh<Osd::CpuGLVertexBuffer,
                             Far::StencilTable,
                             Osd::CpuEvaluator,
                             Osd::GLPatchTable>(
                                 refiner,
                                 numVertexElements,
                                 numVaryingElements,
                                 level, bits);
#ifdef OPENSUBDIV_HAS_OPENMP
    } else if (kernel == "OPENMP") {
        return new Osd::Mesh<Osd::CpuGLVertexBuffer,
                             Far::StencilTable,
                             Osd::OmpEvaluator,
                             Osd::GLPatchTable>(
                                 refiner,
                                 numVertexElements,
                                 numVaryingElements,
                                 level, bits);
#endif
#ifdef OPENSUBDIV_HAS_TBB
    } else if (kernel == "TBB") {
        return new Osd::Mesh<Osd::CpuGLVertexBuffer,
                             Far::StencilTable,
                             Osd::TbbEvaluator,
                             Osd::GLPatchTable>(
                                 refiner,
                                 numVertexElements,
                                 numVaryingElements,
                                 level, bits);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    } else if(kernel == "CL") {
        return new Osd::Mesh<Osd::CLGLVertexBuffer,
                             Osd::CLStencilTable,
                             Osd::CLEvaluator,
                             Osd::GLPatchTable,
                             CLDeviceContext>(
                                 refiner,
                                 numVertexElements,
                                 numVaryingElements,
                                 level, bits,
                                 NULL,
                                 &g_clDeviceContext);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    } else if(kernel == "CUDA") {
        return new Osd::Mesh<Osd::CudaGLVertexBuffer,
                             Osd::CudaStencilTable,
                             Osd::CudaEvaluator,
                             Osd::GLPatchTable>(
                                 refiner,
                                 numVertexElements,
                                 numVaryingElements,
                                 level, bits);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    } else if(kernel == "XFB") {
        return new Osd::Mesh<Osd::GLVertexBuffer,
                             Osd::GLStencilTableTBO,
                             Osd::GLXFBEvaluator,
                             Osd::GLPatchTable>(
                                 refiner,
                                 numVertexElements,
                                 numVaryingElements,
                                 level, bits);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    } else if(kernel == "GLSL") {
        return new Osd::Mesh<Osd::GLVertexBuffer,
                             Osd::GLStencilTableSSBO,
                             Osd::GLComputeEvaluator,
                             Osd::GLPatchTable>(
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
             ShaderCache *shaderCache) {

    std::cout << "Testing " << shapeDesc.name << ", kernel = " << kernel << "\n";

    Shape const * shape = Shape::parseObj(shapeDesc);

    // create Far mesh (topology)
    Sdc::SchemeType sdctype = GetSdcType(*shape);
    Sdc::Options sdcoptions = GetSdcOptions(*shape);

    // create topology refiner
    Far::TopologyRefiner *refiner =
        Far::TopologyRefinerFactory<Shape>::Create(
            *shape,
            Far::TopologyRefinerFactory<Shape>::Options(sdctype, sdcoptions));

    bool interleaveVarying = true;
    bool doSingleCreasePatch = true;

    Osd::MeshBitset bits;
    bits.set(Osd::MeshAdaptive, adaptive);
    bits.set(Osd::MeshUseSingleCreasePatch, doSingleCreasePatch);
    bits.set(Osd::MeshInterleaveVarying, interleaveVarying);
    bits.set(Osd::MeshFVarData, false);
    bits.set(Osd::MeshEndCapGregoryBasis, true);

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
        // normalize xyz
        for (int j = 0; j < 3; ++j) {
            vertex[i*numVertexElements+j] =
                (shape->verts[i*3+j] - center[j])/radius;
        }
        // set rgb from xyz
        for (int j = 0; j < 3; ++j) {
            vertex[i*numVertexElements+j+3] =
                (shape->verts[i*3+j] - pmin[j])*2.0f/radius;
        }
        // set alpha to 1.0
        vertex[i*numVertexElements+3+3] = 1.0f;
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
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh->GetPatchTable()->GetPatchIndexBuffer());
    glBindBuffer(GL_ARRAY_BUFFER, mesh->BindVertexBuffer());

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                          sizeof (GLfloat) * numVertexElements, 0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE,
                          sizeof (GLfloat) * numVertexElements,
                          (const void*)(sizeof(GLfloat)*3));

    // bind patchparam
    if (mesh->GetPatchTable()->GetPatchParamTextureBuffer()) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_BUFFER,
                      mesh->GetPatchTable()->GetPatchParamTextureBuffer());
    }

    Osd::PatchArrayVector const & patches =
        mesh->GetPatchTable()->GetPatchArrays();

    for (int i=0; i<(int)patches.size(); ++i) {
        Osd::PatchArray const & patch = patches[i];
        Far::PatchDescriptor desc = patch.GetDescriptor();
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

        GLuint program = shaderCache->GetDrawConfig(desc)->GetProgram();
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
                               patch.GetPrimitiveIdBase());
        } else {
            glProgramUniform4f(program, diffuseColor, 0.4f, 0.4f, 0.8f, 1);
        }

        glDrawElements(primType,
                       patch.GetNumPatches() * desc.GetNumControlVertices(),
                       GL_UNSIGNED_INT,
                       (void *)(patch.GetIndexBase() * sizeof(unsigned int)));
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
        << "Usage: " << program << "\n"
        << "   -a                      : adaptive refinement (default)\n"
        << "   -u                      : uniform refinement\n"
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
    bool adaptive = true;
    std::string prefix;
    std::string displayMode = "PATCH_TYPE";
    std::vector<std::string> kernels;

    ArgOptions args;

    args.Parse(argc, argv);

    adaptive = args.GetAdaptive();
    isolationLevel = args.GetLevel();

    // Parse remaining args
    const std::vector<const char *> &argvRem = args.GetRemainingArgs();
    for (size_t i = 0; i < argvRem.size(); ++i) {
        if (!strcmp(argvRem[i], "-k")) {
            std::stringstream ss(argvRem[++i]);
            std::string kernel;
            while(std::getline(ss, kernel, ',')) {
                kernels.push_back(kernel);
            }
        } else if (!strcmp(argvRem[i], "-t")) {
            tessLevel = atoi(argvRem[++i]);
        } else if (!strcmp(argvRem[i], "-w")) {
            writeToFile = true;
            prefix = argvRem[++i];
        } else if (!strcmp(argvRem[i], "-s")) {
            width = atoi(argvRem[++i]);
            height = atoi(argvRem[++i]);
        } else if (!strcmp(argvRem[i], "-d")) {
            displayMode = argvRem[++i];
        } else {
            args.PrintUnrecognizedArgWarning(argvRem[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (! glfwInit()) {
        std::cout << "Failed to initialize GLFW\n";
        return 1;
    }

    static const char windowTitle[] =
        "OpenSubdiv imaging test " OPENSUBDIV_VERSION_STRING;

    GLUtils::SetMinimumGLVersion();

    GLFWwindow *window = glfwCreateWindow(width, height, windowTitle, NULL, NULL);
    if (! window) {
        std::cerr << "Failed to create OpenGL context.\n";
        glfwTerminate();
    }

    glfwMakeContextCurrent(window);

    GLUtils::InitializeGL();
    GLUtils::PrintGLVersion();

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
    if (OSD_OPENGL_HAS(VERSION_4_1)) {
        kernels.push_back("XFB");
    }
#endif
#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    if (OSD_OPENGL_HAS(VERSION_4_3)) {
        kernels.push_back("GLSL");
    }
#endif
    }

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
    ShaderCache shaderCache(displayMode);

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

        for (size_t i = 0; i < g_defaultShapes.size(); ++i) {
            ofs << "<tr>\n";
            ofs << "<td>" << g_defaultShapes[i].name << "</td>\n";
            for (size_t k = 0; k < kernels.size(); ++k) {
                ofs << "<td>";
                ofs << "<img src='" << prefix << "_" << kernels[k] << "_" << g_defaultShapes[i].name << ".png'>";
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
        if (kernel == "CL" && CLDeviceContext::HAS_CL_VERSION_1_1()) {
            if (g_clDeviceContext.IsInitialized() == false) {
                if (g_clDeviceContext.Initialize() == false) {
                    std::cout << "Error in initializing OpenCL\n";
                    exit(1);
                }
            }
        }
#endif
#ifdef OPENSUBDIV_HAS_CUDA
        if (kernel == "CUDA") {
            if (g_cudaDeviceContext.IsInitialized() == false) {
                if (g_cudaDeviceContext.Initialize() == false) {
                    std::cout << "Error in initializing Cuda\n";
                    exit(1);
                }
            }
        }
#endif
        for (size_t i = 0; i < g_defaultShapes.size(); ++i) {
            // run test
            runTest(g_defaultShapes[i], kernel, isolationLevel, adaptive, &shaderCache);

            if (writeToFile) {
                // read back pixels
                std::vector<unsigned char> data(width*height*3);
                glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, &data[0]);

                // write image
                std::string filename = prefix + "_" + kernel + "_" + g_defaultShapes[i].name + ".png";
                // flip vertical
                stbi_write_png(filename.c_str(), width, height, 3, &data[width*3*(height-1)], -width*3);
            }

            glfwSwapBuffers(window);
        }
    }

    return 0;
}
