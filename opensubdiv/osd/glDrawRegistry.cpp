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

#include "../osd/glDrawRegistry.h"
#include "../far/error.h"

#include "../osd/opengl.h"

#include <sstream>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

GLDrawConfig::~GLDrawConfig()
{
    glDeleteProgram(program);
}

#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
static const char *commonShaderSource =
#include "glslPatchCommon.gen.h"
;
static const char *bsplineShaderSource =
#include "glslPatchBSpline.gen.h"
;
static const char *gregoryShaderSource =
#include "glslPatchGregory.gen.h"
;
static const char *gregoryBasisShaderSource =
#include "glslPatchGregoryBasis.gen.h"
;
#endif

GLDrawRegistryBase::~GLDrawRegistryBase() {}

#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
GLDrawSourceConfig *
GLDrawRegistryBase::_CreateDrawSourceConfig(Far::PatchDescriptor const & desc)
{
    GLDrawSourceConfig * sconfig = _NewDrawSourceConfig();

    sconfig->commonShader.source = commonShaderSource;

    switch (desc.GetType()) {
    case Far::PatchDescriptor::REGULAR:
        sconfig->commonShader.AddDefine("OSD_PATCH_BSPLINE");
        sconfig->commonShader.AddDefine("OSD_PATCH_ENABLE_SINGLE_CREASE");
        sconfig->vertexShader.source = bsplineShaderSource;
        sconfig->vertexShader.version = "#version 410\n";
        sconfig->vertexShader.AddDefine("OSD_PATCH_VERTEX_BSPLINE_SHADER");
        sconfig->tessControlShader.source = bsplineShaderSource;
        sconfig->tessControlShader.version = "#version 410\n";
        sconfig->tessControlShader.AddDefine("OSD_PATCH_TESS_CONTROL_BSPLINE_SHADER");
        sconfig->tessEvalShader.source = bsplineShaderSource;
        sconfig->tessEvalShader.version = "#version 410\n";
        sconfig->tessEvalShader.AddDefine("OSD_PATCH_TESS_EVAL_BSPLINE_SHADER");
        break;
    case Far::PatchDescriptor::GREGORY:
        sconfig->commonShader.AddDefine("OSD_PATCH_GREGORY");
        sconfig->vertexShader.source = gregoryShaderSource;
        sconfig->vertexShader.version = "#version 410\n";
        sconfig->vertexShader.AddDefine("OSD_PATCH_VERTEX_GREGORY_SHADER");
        sconfig->tessControlShader.source = gregoryShaderSource;
        sconfig->tessControlShader.version = "#version 410\n";
        sconfig->tessControlShader.AddDefine("OSD_PATCH_TESS_CONTROL_GREGORY_SHADER");
        sconfig->tessEvalShader.source = gregoryShaderSource;
        sconfig->tessEvalShader.version = "#version 410\n";
        sconfig->tessEvalShader.AddDefine("OSD_PATCH_TESS_EVAL_GREGORY_SHADER");
        break;
    case Far::PatchDescriptor::GREGORY_BOUNDARY:
        sconfig->commonShader.AddDefine("OSD_PATCH_GREGORY_BOUNDARY");
        sconfig->vertexShader.source = gregoryShaderSource;
        sconfig->vertexShader.version = "#version 410\n";
        sconfig->vertexShader.AddDefine("OSD_PATCH_VERTEX_GREGORY_SHADER");
        sconfig->tessControlShader.source = gregoryShaderSource;
        sconfig->tessControlShader.version = "#version 410\n";
        sconfig->tessControlShader.AddDefine("OSD_PATCH_TESS_CONTROL_GREGORY_SHADER");
        sconfig->tessControlShader.AddDefine("OSD_PATCH_GREGORY_BOUNDARY");
        sconfig->tessEvalShader.source = gregoryShaderSource;
        sconfig->tessEvalShader.version = "#version 410\n";
        sconfig->tessEvalShader.AddDefine("OSD_PATCH_TESS_EVAL_GREGORY_SHADER");
        break;
    case Far::PatchDescriptor::GREGORY_BASIS:
        sconfig->commonShader.AddDefine("OSD_PATCH_GREGORY_BASIS");
        sconfig->vertexShader.source = gregoryBasisShaderSource;
        sconfig->vertexShader.version = "#version 410\n";
        sconfig->vertexShader.AddDefine("OSD_PATCH_VERTEX_GREGORY_BASIS_SHADER");
        sconfig->tessControlShader.source = gregoryBasisShaderSource;
        sconfig->tessControlShader.version = "#version 410\n";
        sconfig->tessControlShader.AddDefine("OSD_PATCH_TESS_CONTROL_GREGORY_BASIS_SHADER");
        sconfig->tessEvalShader.source = gregoryBasisShaderSource;
        sconfig->tessEvalShader.version = "#version 410\n";
        sconfig->tessEvalShader.AddDefine("OSD_PATCH_TESS_EVAL_GREGORY_BASIS_SHADER");
        break;
    default: // POINTS, LINES, QUADS, TRIANGLES
        // do nothing
        break;
    }

    return sconfig;
}
#else
GLDrawSourceConfig *
GLDrawRegistryBase::_CreateDrawSourceConfig(Far::PatchDescriptor const &)
{
    return _NewDrawSourceConfig();
}
#endif

static GLuint
_CompileShader(
        GLenum shaderType,
        OpenSubdiv::Osd::DrawShaderSource const & common,
        OpenSubdiv::Osd::DrawShaderSource const & source)
{
    const char *sources[4];
    std::stringstream definitions;
    for (int i=0; i<(int)common.defines.size(); ++i) {
        definitions << "#define "
                    << common.defines[i].first << " "
                    << common.defines[i].second << "\n";
    }
    for (int i=0; i<(int)source.defines.size(); ++i) {
        definitions << "#define "
                    << source.defines[i].first << " "
                    << source.defines[i].second << "\n";
    }
    std::string defString = definitions.str();

    sources[0] = source.version.c_str();
    sources[1] = defString.c_str();
    sources[2] = common.source.c_str();
    sources[3] = source.source.c_str();

    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 4, sources, NULL);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if( status == GL_FALSE ) {
        GLint infoLogLength;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);
        char * infoLog = new char[infoLogLength];
        glGetShaderInfoLog(shader, infoLogLength, NULL, infoLog);
        Far::Error(Far::FAR_RUNTIME_ERROR,
                   "Error compiling GLSL shader: %s\nDefines: \n%s\n",
                 infoLog, defString.c_str());
        delete[] infoLog;
    }

    return shader;
}

GLDrawConfig *
GLDrawRegistryBase::_CreateDrawConfig(
        Far::PatchDescriptor const & /* desc */,
        GLDrawSourceConfig const * sconfig)
{
    assert(sconfig);

    GLDrawConfig * config = _NewDrawConfig();
    assert(config);

    GLuint vertexShader =
        sconfig->vertexShader.source.empty() ? 0 :
        _CompileShader(GL_VERTEX_SHADER,
                sconfig->commonShader, sconfig->vertexShader);

    GLuint tessControlShader = 0;
    GLuint tessEvalShader = 0;

#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
    tessControlShader =
        sconfig->tessControlShader.source.empty() ? 0 :
        _CompileShader(GL_TESS_CONTROL_SHADER,
                sconfig->commonShader, sconfig->tessControlShader);

    tessEvalShader =
        sconfig->tessEvalShader.source.empty() ? 0 :
        _CompileShader(GL_TESS_EVALUATION_SHADER,
                sconfig->commonShader, sconfig->tessEvalShader);
#endif

    GLuint geometryShader = 0;

#if defined(GL_ARB_geometry_shader4) || defined(GL_VERSION_3_1)
    geometryShader =
        sconfig->geometryShader.source.empty() ? 0 :
        _CompileShader(GL_GEOMETRY_SHADER,
                sconfig->commonShader, sconfig->geometryShader);
#endif

    GLuint fragmentShader =
        sconfig->fragmentShader.source.empty() ? 0 :
        _CompileShader(GL_FRAGMENT_SHADER,
                sconfig->commonShader, sconfig->fragmentShader);

    GLuint program = glCreateProgram();
    if (vertexShader)      glAttachShader(program, vertexShader);
    if (tessControlShader) glAttachShader(program, tessControlShader);
    if (tessEvalShader)    glAttachShader(program, tessEvalShader);
    if (geometryShader)    glAttachShader(program, geometryShader);
    if (fragmentShader)    glAttachShader(program, fragmentShader);

    glLinkProgram(program);

    glDeleteShader(vertexShader);
    glDeleteShader(tessControlShader);
    glDeleteShader(tessEvalShader);
    glDeleteShader(geometryShader);
    glDeleteShader(fragmentShader);

    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status );
    if( status == GL_FALSE ) {
        GLint infoLogLength;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);
        char * infoLog = new char[infoLogLength];
        glGetProgramInfoLog(program, infoLogLength, NULL, infoLog);
        Far::Error(Far::FAR_RUNTIME_ERROR,
                   "Error linking GLSL program: %s\n", infoLog);
        delete[] infoLog;
    }

    config->program = program;

    return config;
}

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
