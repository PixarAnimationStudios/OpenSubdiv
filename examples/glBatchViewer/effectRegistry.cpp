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
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
//
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

#include "effectRegistry.h"

static const char *shaderSource =
#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
    #include "shader.inc"
#else
    #include "shader_gl3.inc"
#endif
;


MyEffectRegistry::SourceConfigType *
MyEffectRegistry::_CreateDrawSourceConfig(DescType const & desc) {

    SourceConfigType * sconfig =
        BaseRegistry::_CreateDrawSourceConfig(desc.first);
    assert(sconfig);

    EffectDesc effectDesc = desc.second;

    if (effectDesc.GetScreenSpaceTess()) {
        sconfig->commonShader.AddDefine("OSD_ENABLE_PATCH_CULL");
        sconfig->commonShader.AddDefine("OSD_ENABLE_SCREENSPACE_TESSELLATION");
    }

    const char *glslVersion = "#version 400\n";
    if (desc.first.GetType() == OpenSubdiv::FarPatchTables::QUADS) {
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
        sconfig->geometryShader.AddDefine("PRIM_QUAD");
        sconfig->fragmentShader.AddDefine("PRIM_QUAD");
    } else {
        sconfig->geometryShader.AddDefine("PRIM_TRI");
        sconfig->fragmentShader.AddDefine("PRIM_TRI");
    }

    int wire = effectDesc.GetWire();

    if (wire == MyEffect::kWire) {
        sconfig->geometryShader.AddDefine("GEOMETRY_OUT_WIRE");
        sconfig->fragmentShader.AddDefine("GEOMETRY_OUT_WIRE");
    } else if (wire == MyEffect::kFill) {
        sconfig->geometryShader.AddDefine("GEOMETRY_OUT_FILL");
        sconfig->fragmentShader.AddDefine("GEOMETRY_OUT_FILL");
    } else if (wire == MyEffect::kLine) {
        sconfig->geometryShader.AddDefine("GEOMETRY_OUT_LINE");
        sconfig->fragmentShader.AddDefine("GEOMETRY_OUT_LINE");
    }

    return sconfig;

}

MyEffectRegistry::ConfigType *
MyEffectRegistry::_CreateDrawConfig(DescType const & desc, SourceConfigType const * sconfig) {

    // XXX: make sure is this downcast correct
    ConfigType * config = (ConfigType *)BaseRegistry::_CreateDrawConfig(desc.first, sconfig);
    assert(config);

    GLuint uboIndex;
    GLuint program = config->program;

    GLuint g_transformBinding = 0,
        g_tessellationBinding = 0,
        g_lightingBinding = 0;

    // XXXdyu can use layout(binding=) with GLSL 4.20 and beyond
    g_transformBinding = 0;
    uboIndex = glGetUniformBlockIndex(program, "Transform");
    if (uboIndex != GL_INVALID_INDEX)
        glUniformBlockBinding(program, uboIndex, g_transformBinding);

    g_tessellationBinding = 1;
    uboIndex = glGetUniformBlockIndex(program, "Tessellation");
    if (uboIndex != GL_INVALID_INDEX)
        glUniformBlockBinding(program, uboIndex, g_tessellationBinding);

    g_lightingBinding = 3;
    uboIndex = glGetUniformBlockIndex(program, "Lighting");
    if (uboIndex != GL_INVALID_INDEX)
        glUniformBlockBinding(program, uboIndex, g_lightingBinding);

//    g_gregoryQuadOffsetBaseMap[program] = glGetUniformLocation(program, "GregoryQuadOffsetBase");
//    g_levelBaseMap[program] = glGetUniformLocation(program, "LevelBase");

// currently, these are used only in conjunction with tessellation shaders
#if defined(GL_EXT_direct_state_access) || defined(GL_VERSION_4_1)
    GLint loc;
    if ((loc = glGetUniformLocation(program, "g_VertexBuffer")) != -1) {
        glProgramUniform1i(program, loc, 0); // GL_TEXTURE0
    }
    if ((loc = glGetUniformLocation(program, "g_ValenceBuffer")) != -1) {
        glProgramUniform1i(program, loc, 1); // GL_TEXTURE1
    }
    if ((loc = glGetUniformLocation(program, "g_QuadOffsetBuffer")) != -1) {
        glProgramUniform1i(program, loc, 2); // GL_TEXTURE2
    }
    if ((loc = glGetUniformLocation(program, "g_ptexIndicesBuffer")) != -1) {
        glProgramUniform1i(program, loc, 3); // GL_TEXTURE3
    }
#endif

    config->diffuseColorUniform = glGetUniformLocation(program, "diffuseColor");

    return config;
}
