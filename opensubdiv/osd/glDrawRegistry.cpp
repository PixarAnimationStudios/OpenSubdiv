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

#if defined(__APPLE__)
    #include "TargetConditionals.h"
    #if TARGET_OS_IPHONE or TARGET_IPHONE_SIMULATOR
        #include <OpenGLES/ES2/gl.h>
    #else
        #include <OpenGL/gl3.h>
    #endif
#elif defined(ANDROID)
    #include <GLES2/gl2.h>
#else
    #if defined(_WIN32)
        #include <windows.h>
    #endif
    #include <GL/glew.h>
#endif

#include "../osd/glDrawRegistry.h"
#include "../osd/error.h"

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
static const char *regularShaderSource =
#include "glslPatchRegular.inc"
;
static const char *boundaryShaderSource =
#include "glslPatchBoundary.inc"
;
static const char *cornerShaderSource =
#include "glslPatchCorner.inc"
;
static const char *gregoryShaderSource =
#include "glslPatchGregory.inc"
;
static const char *boundaryGregoryShaderSource =
#include "glslPatchBoundaryGregory.inc"
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
            sconfig->vertexShader.source = regularShaderSource;
            sconfig->vertexShader.version = "#version 410\n";
            sconfig->vertexShader.AddDefine("VERTEX_SHADER");
            sconfig->fragmentShader.source = regularShaderSource;
            sconfig->fragmentShader.AddDefine("FRAGMENT_SHADER");
            sconfig->fragmentShader.version = "#version 410\n";
            break;
        case FarPatchTables::REGULAR:
            sconfig->vertexShader.source = regularShaderSource;
            sconfig->vertexShader.version = "#version 410\n";
            sconfig->vertexShader.AddDefine("PATCH_VERTEX_SHADER");
            sconfig->tessControlShader.source = regularShaderSource;
            sconfig->tessControlShader.version = "#version 410\n";
            sconfig->tessControlShader.AddDefine("PATCH_TESS_CONTROL_REGULAR_SHADER");
            sconfig->tessEvalShader.source = regularShaderSource;
            sconfig->tessEvalShader.version = "#version 410\n";
            sconfig->tessEvalShader.AddDefine("PATCH_TESS_EVAL_REGULAR_SHADER");
            sconfig->fragmentShader.source = regularShaderSource;
            sconfig->fragmentShader.version = "#version 410\n";
            sconfig->fragmentShader.AddDefine("FRAGMENT_SHADER");
            break;
        case FarPatchTables::BOUNDARY:
            sconfig->vertexShader.source = boundaryShaderSource;
            sconfig->vertexShader.version = "#version 410\n";
            sconfig->vertexShader.AddDefine("PATCH_VERTEX_SHADER");
            sconfig->tessControlShader.source = boundaryShaderSource;
            sconfig->tessControlShader.version = "#version 410\n";
            sconfig->tessControlShader.AddDefine("PATCH_TESS_CONTROL_BOUNDARY_SHADER");
            sconfig->tessEvalShader.source = boundaryShaderSource;
            sconfig->tessEvalShader.version = "#version 410\n";
            sconfig->tessEvalShader.AddDefine("PATCH_TESS_EVAL_BOUNDARY_SHADER");
            sconfig->fragmentShader.source = boundaryShaderSource;
            sconfig->fragmentShader.version = "#version 410\n";
            sconfig->fragmentShader.AddDefine("FRAGMENT_SHADER");
            break;
        case FarPatchTables::CORNER:
            sconfig->vertexShader.source = cornerShaderSource;
            sconfig->vertexShader.version = "#version 410\n";
            sconfig->vertexShader.AddDefine("PATCH_VERTEX_SHADER");
            sconfig->tessControlShader.source = cornerShaderSource;
            sconfig->tessControlShader.version = "#version 410\n";
            sconfig->tessControlShader.AddDefine("PATCH_TESS_CONTROL_CORNER_SHADER");
            sconfig->tessEvalShader.source = cornerShaderSource;
            sconfig->tessEvalShader.version = "#version 410\n";
            sconfig->tessEvalShader.AddDefine("PATCH_TESS_EVAL_CORNER_SHADER");
            sconfig->fragmentShader.source = cornerShaderSource;
            sconfig->fragmentShader.version = "#version 410\n";
            sconfig->fragmentShader.AddDefine("FRAGMENT_SHADER");
            break;
        case FarPatchTables::GREGORY:
            sconfig->vertexShader.source = gregoryShaderSource;
            sconfig->vertexShader.version = "#version 410\n";
            sconfig->vertexShader.AddDefine("PATCH_VERTEX_GREGORY_SHADER");
            sconfig->tessControlShader.source = gregoryShaderSource;
            sconfig->tessControlShader.version = "#version 410\n";
            sconfig->tessControlShader.AddDefine("PATCH_TESS_CONTROL_GREGORY_SHADER");
            sconfig->tessEvalShader.source = gregoryShaderSource;
            sconfig->tessEvalShader.version = "#version 410\n";
            sconfig->tessEvalShader.AddDefine("PATCH_TESS_EVAL_GREGORY_SHADER");
            sconfig->fragmentShader.source = gregoryShaderSource;
            sconfig->fragmentShader.version = "#version 410\n";
            sconfig->fragmentShader.AddDefine("FRAGMENT_SHADER");
            break;
        case FarPatchTables::GREGORY_BOUNDARY:
            sconfig->vertexShader.source = boundaryGregoryShaderSource;
            sconfig->vertexShader.version = "#version 410\n";
            sconfig->vertexShader.AddDefine("PATCH_VERTEX_BOUNDARY_GREGORY_SHADER");
            sconfig->tessControlShader.source = boundaryGregoryShaderSource;
            sconfig->tessControlShader.version = "#version 410\n";
            sconfig->tessControlShader.AddDefine("PATCH_TESS_CONTROL_BOUNDARY_GREGORY_SHADER");
            sconfig->tessEvalShader.source = boundaryGregoryShaderSource;
            sconfig->tessEvalShader.version = "#version 410\n";
            sconfig->tessEvalShader.AddDefine("PATCH_TESS_EVAL_BOUNDARY_GREGORY_SHADER");
            sconfig->fragmentShader.source = boundaryGregoryShaderSource;
            sconfig->fragmentShader.version = "#version 410\n";
            sconfig->fragmentShader.AddDefine("FRAGMENT_SHADER");
            break;
        default:
            // error
            delete sconfig;
            sconfig = NULL;
            break;
        }
    } else { // pattern != NON_TRANSITION
        sconfig->vertexShader.source = transitionShaderSource;
        sconfig->vertexShader.version = "#version 410\n";
        sconfig->vertexShader.AddDefine("PATCH_VERTEX_SHADER");
        sconfig->tessControlShader.source = transitionShaderSource;;
        sconfig->tessControlShader.version = "#version 410\n";
        sconfig->tessControlShader.AddDefine("PATCH_TESS_CONTROL_TRANSITION_SHADER");
        sconfig->tessEvalShader.source = transitionShaderSource;
        sconfig->tessEvalShader.version = "#version 410\n";
        sconfig->tessEvalShader.AddDefine("PATCH_TESS_EVAL_TRANSITION_SHADER");
        sconfig->fragmentShader.source = transitionShaderSource;
        sconfig->fragmentShader.version = "#version 410\n";
        sconfig->fragmentShader.AddDefine("FRAGMENT_SHADER");

        int pattern = desc.GetPattern() - 1;
        int rotation = desc.GetRotation();
        int subpatch = desc.GetSubPatch();

        std::ostringstream ss;
        ss << "CASE" << pattern << subpatch;
        sconfig->tessControlShader.AddDefine(ss.str());
        sconfig->tessEvalShader.AddDefine(ss.str());

        ss.str("");
        ss << rotation;
        sconfig->tessControlShader.AddDefine("ROTATE", ss.str());
        sconfig->tessEvalShader.AddDefine("ROTATE", ss.str());

        if (desc.GetType() == FarPatchTables::BOUNDARY) {
            sconfig->tessControlShader.AddDefine("BOUNDARY");
        } else if (desc.GetType() == FarPatchTables::CORNER) {
            sconfig->tessControlShader.AddDefine("CORNER");
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
        OsdError(OSD_GLSL_COMPILE_ERROR,
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
