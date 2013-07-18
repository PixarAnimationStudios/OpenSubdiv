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

#include "../osd/glDrawRegistry.h"
#include "../osd/error.h"

#include "../osd/opengl.h"

#include <sstream>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdGLDrawConfig::~OsdGLDrawConfig()
{
    glDeleteProgram(program);
}

#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
static const char *commonShaderSource =
#include "glslPatchCommon.inc"
;
static const char *bsplineShaderSource =
#include "glslPatchBSpline.inc"
;
static const char *gregoryShaderSource =
#include "glslPatchGregory.inc"
;
static const char *transitionShaderSource =
#include "glslPatchTransition.inc"
;
#endif

OsdGLDrawRegistryBase::~OsdGLDrawRegistryBase() {}

OsdGLDrawSourceConfig *
OsdGLDrawRegistryBase::_CreateDrawSourceConfig(OsdDrawContext::PatchDescriptor const & desc)
{
    OsdGLDrawSourceConfig * sconfig = _NewDrawSourceConfig();

#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
    sconfig->commonShader.source = commonShaderSource;
    {
        std::ostringstream ss;
        ss << (int)desc.GetMaxValence();
        sconfig->commonShader.AddDefine("OSD_MAX_VALENCE", ss.str());
        ss.str("");
        ss << (int)desc.GetNumElements();
        sconfig->commonShader.AddDefine("OSD_NUM_ELEMENTS", ss.str());
    }

    if (desc.GetPattern() == FarPatchTables::NON_TRANSITION) {
        switch (desc.GetType()) {
        case FarPatchTables::QUADS:
        case FarPatchTables::TRIANGLES:
            // do nothing
            break;
        case FarPatchTables::REGULAR:
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
        case FarPatchTables::BOUNDARY:
            sconfig->vertexShader.source = bsplineShaderSource;
            sconfig->vertexShader.version = "#version 410\n";
            sconfig->vertexShader.AddDefine("OSD_PATCH_VERTEX_BSPLINE_SHADER");
            sconfig->tessControlShader.source = bsplineShaderSource;
            sconfig->tessControlShader.version = "#version 410\n";
            sconfig->tessControlShader.AddDefine("OSD_PATCH_TESS_CONTROL_BSPLINE_SHADER");
            sconfig->tessControlShader.AddDefine("OSD_PATCH_BOUNDARY");
            sconfig->tessEvalShader.source = bsplineShaderSource;
            sconfig->tessEvalShader.version = "#version 410\n";
            sconfig->tessEvalShader.AddDefine("OSD_PATCH_TESS_EVAL_BSPLINE_SHADER");
            break;
        case FarPatchTables::CORNER:
            sconfig->vertexShader.source = bsplineShaderSource;
            sconfig->vertexShader.version = "#version 410\n";
            sconfig->vertexShader.AddDefine("OSD_PATCH_VERTEX_BSPLINE_SHADER");
            sconfig->tessControlShader.source = bsplineShaderSource;
            sconfig->tessControlShader.version = "#version 410\n";
            sconfig->tessControlShader.AddDefine("OSD_PATCH_TESS_CONTROL_BSPLINE_SHADER");
            sconfig->tessControlShader.AddDefine("OSD_PATCH_CORNER");
            sconfig->tessEvalShader.source = bsplineShaderSource;
            sconfig->tessEvalShader.version = "#version 410\n";
            sconfig->tessEvalShader.AddDefine("OSD_PATCH_TESS_EVAL_BSPLINE_SHADER");
            break;
        case FarPatchTables::GREGORY:
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
        case FarPatchTables::GREGORY_BOUNDARY:
            sconfig->vertexShader.source = gregoryShaderSource;
            sconfig->vertexShader.version = "#version 410\n";
            sconfig->vertexShader.AddDefine("OSD_PATCH_VERTEX_GREGORY_SHADER");
            sconfig->vertexShader.AddDefine("OSD_PATCH_GREGORY_BOUNDARY");
            sconfig->tessControlShader.source = gregoryShaderSource;
            sconfig->tessControlShader.version = "#version 410\n";
            sconfig->tessControlShader.AddDefine("OSD_PATCH_TESS_CONTROL_GREGORY_SHADER");
            sconfig->tessControlShader.AddDefine("OSD_PATCH_GREGORY_BOUNDARY");
            sconfig->tessEvalShader.source = gregoryShaderSource;
            sconfig->tessEvalShader.version = "#version 410\n";
            sconfig->tessEvalShader.AddDefine("OSD_PATCH_TESS_EVAL_GREGORY_SHADER");
            sconfig->tessEvalShader.AddDefine("OSD_PATCH_GREGORY_BOUNDARY");
            break;
        default:
            // error
            delete sconfig;
            sconfig = NULL;
            break;
        }
    } else { // pattern != NON_TRANSITION
        sconfig->vertexShader.source = bsplineShaderSource;
        sconfig->vertexShader.version = "#version 410\n";
        sconfig->vertexShader.AddDefine("OSD_PATCH_VERTEX_BSPLINE_SHADER");
        sconfig->tessControlShader.source =
                std::string(transitionShaderSource) + bsplineShaderSource;
        sconfig->tessControlShader.version = "#version 410\n";
        sconfig->tessControlShader.AddDefine("OSD_PATCH_TESS_CONTROL_BSPLINE_SHADER");
        sconfig->tessControlShader.AddDefine("OSD_PATCH_TRANSITION");
        sconfig->tessEvalShader.source =
                std::string(transitionShaderSource) + bsplineShaderSource;
        sconfig->tessEvalShader.version = "#version 410\n";
        sconfig->tessEvalShader.AddDefine("OSD_PATCH_TESS_EVAL_BSPLINE_SHADER");
        sconfig->tessEvalShader.AddDefine("OSD_PATCH_TRANSITION");

        int pattern = desc.GetPattern() - 1;
        int rotation = desc.GetRotation();
        int subpatch = desc.GetSubPatch();

        std::ostringstream ss;
        ss << "OSD_TRANSITION_PATTERN" << pattern << subpatch;
        sconfig->tessControlShader.AddDefine(ss.str());
        sconfig->tessEvalShader.AddDefine(ss.str());

        ss.str("");
        ss << rotation;
        sconfig->tessControlShader.AddDefine("OSD_TRANSITION_ROTATE", ss.str());
        sconfig->tessEvalShader.AddDefine("OSD_TRANSITION_ROTATE", ss.str());

        if (desc.GetType() == FarPatchTables::BOUNDARY) {
            sconfig->tessControlShader.AddDefine("OSD_PATCH_BOUNDARY");
        } else if (desc.GetType() == FarPatchTables::CORNER) {
            sconfig->tessControlShader.AddDefine("OSD_PATCH_CORNER");
        }
    }
#endif

    return sconfig;
}

static GLuint
_CompileShader(
        GLenum shaderType,
        OpenSubdiv::OsdDrawShaderSource const & common,
        OpenSubdiv::OsdDrawShaderSource const & source)
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
        OsdError(OSD_GLSL_COMPILE_ERROR,
                 "Error compiling GLSL shader: %s\nDefines: \n%s\n",
                 infoLog, defString.c_str());
        delete[] infoLog;
    }

    return shader;
}

OsdGLDrawConfig *
OsdGLDrawRegistryBase::_CreateDrawConfig(
        OsdDrawContext::PatchDescriptor const & desc,
        OsdGLDrawSourceConfig const * sconfig) 
{
    assert(sconfig);

    OsdGLDrawConfig * config = _NewDrawConfig();
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
        OsdError(OSD_GLSL_LINK_ERROR,
                 "Error linking GLSL program: %s\n", infoLog);
        delete[] infoLog;
    }

    config->program = program;
    config->primitiveIdBaseUniform =
      glGetUniformLocation(program, "OsdPrimitiveIdBase");
    config->gregoryQuadOffsetBaseUniform =
      glGetUniformLocation(program, "OsdGregoryQuadOffsetBase");

    return config;
}

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
