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
    config->levelBaseUniform = glGetUniformLocation(program, "LevelBase");
    config->gregoryQuadOffsetBaseUniform = glGetUniformLocation(program, "GregoryQuadOffsetBase");

    return config;
}

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
